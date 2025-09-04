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

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight, gather_params_ctx
from peft.utils.other import transpose

# Import FourierFT config
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../fourierft'))
from config import FourierFTConfig


class FourierFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("fourierft_spectrum",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("fourierft_n_frequency", "fourierft_scaling", "fourierft_random_loc_seed")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.fourierft_n_frequency = {}
        self.fourierft_scaling = {}
        self.fourierft_spectrum = nn.ParameterDict({})
        self.indices = {}
        self.fourierft_random_loc_seed = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

    def update_layer(self, adapter_name, n_frequency, scaling, random_loc_seed, init_fourierft_weights):
        # Store FourierFT parameters
        self.fourierft_n_frequency[adapter_name] = n_frequency
        self.fourierft_scaling[adapter_name] = scaling
        self.fourierft_random_loc_seed[adapter_name] = random_loc_seed
        
        # Create FourierFT spectrum parameter
        self.fourierft_spectrum[adapter_name] = nn.Parameter(
            torch.randn(n_frequency, dtype=torch.float32)
        )
        
        # Generate indices for FourierFT
        self._generate_indices(adapter_name)
        
        if init_fourierft_weights:
            self.reset_fourierft_parameters(adapter_name)

    def _generate_indices(self, adapter_name):
        """Generate indices for FourierFT frequency selection"""
        n_frequency = self.fourierft_n_frequency[adapter_name]
        random_loc_seed = self.fourierft_random_loc_seed[adapter_name]
        
        # Generate random indices based on seed
        torch.manual_seed(random_loc_seed)
        total_elements = self.in_features * self.out_features
        indices = torch.randperm(total_elements)[:n_frequency]
        
        # Convert to 2D indices
        row_indices = indices // self.out_features
        col_indices = indices % self.out_features
        
        self.indices[adapter_name] = (row_indices, col_indices)

    def reset_fourierft_parameters(self, adapter_name):
        """Initialize FourierFT spectrum parameters"""
        if adapter_name in self.fourierft_spectrum.keys():
            # Initialize spectrum with small random values
            nn.init.normal_(self.fourierft_spectrum[adapter_name], std=0.02)

    def get_delta_weight(self, adapter_name) -> torch.Tensor:
        """Compute delta weight from FourierFT spectrum"""
        if adapter_name not in self.fourierft_spectrum:
            return torch.zeros((self.out_features, self.in_features), device=self.base_layer.weight.device)
        
        spectrum = self.fourierft_spectrum[adapter_name]
        scaling = self.fourierft_scaling[adapter_name]
        row_indices, col_indices = self.indices[adapter_name]
        
        # Create delta weight matrix
        delta_w = torch.zeros((self.out_features, self.in_features), 
                             device=spectrum.device, dtype=spectrum.dtype)
        
        # Apply FourierFT transformation (simplified version)
        # In practice, this would involve more complex Fourier operations
        delta_w[row_indices, col_indices] = spectrum * scaling
        
        return delta_w


class Linear(nn.Module, FourierFTLayer):
    # FourierFT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 1000,
        scaling: float = 1.0,
        random_loc_seed: int = 42,
        init_fourierft_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierFTLayer.__init__(self, base_layer, **kwargs)

        self.update_layer(adapter_name, n_frequency, scaling, random_loc_seed, init_fourierft_weights)
        self._active_adapter = adapter_name
    
    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def _check_forward_args(self, x, *args, **kwargs):
        """Check that forward arguments are valid"""
        pass  # Basic implementation, can be extended if needed

    def _mixed_batch_forward(self, x, *args, adapter_names=None, **kwargs):
        """Handle mixed batch forward (basic implementation)"""
        return self.base_layer(x, *args, **kwargs), torch.tensor(0.0, device=x.device)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.fourierft_spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.fourierft_spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def forward(self, x: torch.Tensor, active_adapters_d, ood_weight, o_fourierft_layer, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass with O3 soft-weighted inference and orthogonal loss
        This maintains the same interface as lora_layer_hacked_o.py
        """
        previous_dtype = x.dtype
        adapter_names = kwargs.pop("adapter_names", None)
        
        # Initialize orthogonal loss
        o_loss = torch.tensor(0.0, device=x.device)

        if self._disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            # **CRITICAL FIX: Use active_adapters_d parameter like LoRA**
            for active_adapter in active_adapters_d:  # This ensures we use the current training adapter
                if active_adapter not in self.fourierft_spectrum.keys():
                    continue
                
                # **Orthogonal loss computation FIRST (like LoRA implementation)**
                if o_fourierft_layer is not None and len(o_fourierft_layer) > 0:
                    # Get current spectrum (the actual trainable parameters in FourierFT)
                    current_spectrum = self.fourierft_spectrum[active_adapter]
                    
                    for i in range(len(o_fourierft_layer)):
                        # o_fourierft_layer[i] contains previous spectrum vectors
                        # Compute orthogonal loss: ensure current spectrum is orthogonal to previous ones
                        dot_product = torch.dot(current_spectrum.flatten(), o_fourierft_layer[i].flatten())
                        o_loss = o_loss + dot_product ** 2

                # Get FourierFT delta weight
                delta_w = self.get_delta_weight(active_adapter)
                
                # Compute FourierFT output
                x_casted = x.to(delta_w.dtype)
                fourierft_out = F.linear(x_casted, delta_w)
                
                # **KEY MODIFICATION: O3 soft-weighted inference**
                # This is the same as LoRA version: result * α₁ + adapter_out * α₂
                result = result * ood_weight[0] + fourierft_out * ood_weight[1]

            result = result.to(torch_result_dtype)

        return result, o_loss

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourierft." + rep


class Embedding(nn.Module, FourierFTLayer):
    # FourierFT implemented in an Embedding layer
    def __init__(
        self,
        base_layer: nn.Embedding,
        adapter_name: str,
        n_frequency: int = 1000,
        scaling: float = 1.0,
        random_loc_seed: int = 42,
        init_fourierft_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierFTLayer.__init__(self, base_layer, **kwargs)

        self.update_layer(adapter_name, n_frequency, scaling, random_loc_seed, init_fourierft_weights)
        self._active_adapter = adapter_name

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.fourierft_spectrum.keys():
                    continue

                # Apply FourierFT to embedding
                delta_w = self.get_delta_weight(active_adapter)
                after_A = F.embedding(x, delta_w, self.base_layer.padding_idx, 
                                    self.base_layer.max_norm, self.base_layer.norm_type,
                                    self.base_layer.scale_grad_by_freq, self.base_layer.sparse)
                result = result + after_A

            result = result.to(torch_result_dtype)
        return result
