# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional, Union
from itertools import chain
from tqdm import tqdm

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

# Import FourierFT config and layer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../fourierft'))
from config import FourierFTConfig
from src.fourierft_layer_hacked_o import FourierFTLayer, Linear, Embedding

if is_bnb_available():
    import bitsandbytes as bnb

    from peft.tuners.lora.bnb import Linear4bit, Linear8bitLt

if is_bnb_4bit_available():
    from peft.tuners.lora.bnb import Linear4bit


class FourierFTModel(BaseTuner):
    """
    Creates FourierFT model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`FourierFTConfig`]): The configuration of the FourierFT model.

    Returns:
        `torch.nn.Module`: The FourierFT model.

    Example:
        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import FourierFTModel, FourierFTConfig

        >>> config = FourierFTConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     n_frequency=1000,
        ...     scaling=150.0,
        ...     target_modules=["q", "v"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> fourierft_model = FourierFTModel(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`FourierFTConfig`]): The configuration of the FourierFT model.
    """

    prefix: str = "fourierft_"
    layers_mapping = {
        nn.Linear: Linear,
        nn.Embedding: Embedding,
        nn.Conv2d: Linear,  # Treat Conv2d as Linear for FourierFT
    }

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: FourierFTConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.
        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(fourierft_config, key):
        return check_target_module_exists(fourierft_config, key)

    def _create_and_replace(
        self,
        fourierft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find the layer
        pattern_keys = list(chain(fourierft_config.n_frequency_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        n_frequency = fourierft_config.n_frequency_pattern.get(target_name_key, fourierft_config.n_frequency)

        if isinstance(target, FourierFTLayer):
            target.update_layer(
                adapter_name,
                n_frequency,
                fourierft_config.scaling,
                fourierft_config.random_loc_seed,
                fourierft_config.init_weights,
            )
        else:
            new_module = self._create_new_module(fourierft_config, adapter_name, target)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)

    def _create_new_module(self, fourierft_config, adapter_name, target):
        # Avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from peft.tuners.lora.bnb import Linear4bit, Linear8bitLt

        if is_bnb_4bit_available():
            from peft.tuners.lora.bnb import Linear4bit

        loaded_in_kbit = getattr(self.model, "is_loaded_in_8bit", False) or getattr(
            self.model, "is_loaded_in_4bit", False
        )
        if loaded_in_kbit and not is_bnb_available():
            raise ImportError(
                "To use FourierFT with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        # Get FourierFT parameters from config
        n_frequency = fourierft_config.n_frequency
        scaling = fourierft_config.scaling
        random_loc_seed = fourierft_config.random_loc_seed
        init_fourierft_weights = fourierft_config.init_weights

        if loaded_in_kbit:
            if is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
                fourierft_layer = Linear4bit(target, adapter_name, n_frequency=n_frequency, scaling=scaling, 
                                           random_loc_seed=random_loc_seed, init_fourierft_weights=init_fourierft_weights)
            elif isinstance(target_base_layer, bnb.nn.Linear8bitLt):
                fourierft_layer = Linear8bitLt(target, adapter_name, n_frequency=n_frequency, scaling=scaling,
                                             random_loc_seed=random_loc_seed, init_fourierft_weights=init_fourierft_weights)
            else:
                raise ValueError(
                    f"Target module {target} is not supported. Currently, only `torch.nn.Linear`, "
                    "`torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`, "
                    "`bnb.nn.Linear4bit`, `bnb.nn.Linear8bitLt` are supported."
                )
        elif isinstance(target_base_layer, torch.nn.Linear):
            fourierft_layer = Linear(target, adapter_name, n_frequency=n_frequency, scaling=scaling,
                                   random_loc_seed=random_loc_seed, init_fourierft_weights=init_fourierft_weights)
        elif isinstance(target_base_layer, torch.nn.Embedding):
            fourierft_layer = Embedding(target, adapter_name, n_frequency=n_frequency, scaling=scaling,
                                      random_loc_seed=random_loc_seed, init_fourierft_weights=init_fourierft_weights)
        elif isinstance(target_base_layer, torch.nn.Conv2d):
            fourierft_layer = Linear(target, adapter_name, n_frequency=n_frequency, scaling=scaling,
                                   random_loc_seed=random_loc_seed, init_fourierft_weights=init_fourierft_weights)  # Use Linear for Conv2d
        elif isinstance(target_base_layer, Conv1D):
            fourierft_layer = Linear(target, adapter_name, n_frequency=n_frequency, scaling=scaling,
                                   random_loc_seed=random_loc_seed, init_fourierft_weights=init_fourierft_weights)  # Use Linear for Conv1D
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear`, "
                "`torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D` are supported."
            )

        return fourierft_layer

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, FourierFTLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        self._unloading_checks(adapter_names)
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, FourierFTLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        r"""
        This method merges the FourierFT layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Mark only the FourierFT adapter parameters as trainable."""
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)