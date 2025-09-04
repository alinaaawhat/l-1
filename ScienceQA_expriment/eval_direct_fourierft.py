#!/usr/bin/env python3

import os
import sys
import json
from typing import Optional
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import LlamaTokenizer
from tqdm import tqdm

# Add FourierFT path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../fourierft'))
from config import FourierFTConfig

from src.modeling_llama_hacked_o import LlamaForCausalLM_ood
from src.peft_model_hacked_o import PeftModel
from utils.prompter import Prompter

def evaluate_fourierft_direct(
    test_dataset: str = "./data/scienceqa_RD_5/scienceqa_not_biology_test_RD.json",
    base_model: str = "gcyzsl/O3_LLAMA2_ScienceQA",
    fourierft_weights: str = "",  # Path to FourierFT adapter
    seed: int = 0,
    prompt_template_name: str = "alpaca",
    max_length: int = 256,
    batch_size: int = 8,
):
    """
    Direct evaluation of FourierFT orthogonal training results without soft inference.
    Uses fixed ood_weight=[1,1] to evaluate pure FourierFT adapter performance.
    """
    
    print(f"=== Direct FourierFT Evaluation ===")
    print(f"Base model: {base_model}")
    print(f"FourierFT weights: {fourierft_weights}")
    print(f"Test dataset: {test_dataset}")
    print(f"Fixed ood_weight: [1, 1] (no soft inference)")
    print("=" * 50)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Initialize with fixed ood_weight (no soft inference)
    model.init_oodweight(ood_weight=[1, 1])  # Equal weight to base and adapter
    model.init_active_adapters_d(active_adapters_d=['default'])
    model.init_ofourierft(orthogonal_loss=False, ofourierft_weights=[])  # No orthogonal loss during eval
    
    # Load FourierFT adapter if provided
    if fourierft_weights and os.path.exists(fourierft_weights):
        print(f"Loading FourierFT adapter from: {fourierft_weights}")
        model = PeftModel.from_pretrained(model, fourierft_weights, is_trainable=False)
    else:
        print("No FourierFT adapter loaded - evaluating base model only")

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    # Load prompter
    prompter = Prompter(prompt_template_name)
    
    # Load test dataset
    if test_dataset.endswith('.json'):
        data = load_dataset("json", data_files=test_dataset)['train']
    else:
        raise ValueError("Only JSON format is supported")
    
    model.eval()
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(data)} samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(data)):
            try:
                # Generate prompt (without output for evaluation)
                prompt = prompter.generate_prompt(
                    sample["instruction"],
                    sample.get("input", "")
                )
                
                # Tokenize input
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True
                ).to(device)
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode response
                generated_text = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Extract answer (simple matching)
                expected_output = sample.get("output", "").strip().lower()
                generated_output = generated_text.lower()
                
                # Simple accuracy check (you may need to adjust this based on your data format)
                if expected_output in generated_output or generated_output.startswith(expected_output[:3]):
                    correct += 1
                
                total += 1
                
                # Print progress every 100 samples
                if (i + 1) % 100 == 0:
                    current_acc = correct / total
                    print(f"Progress: {i+1}/{len(data)}, Current Accuracy: {current_acc:.4f}")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Save results
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "model_path": base_model,
        "adapter_path": fourierft_weights,
        "test_dataset": test_dataset,
        "ood_weight": [1, 1],
        "evaluation_type": "direct_fourierft"
    }
    
    # Create output filename
    output_file = test_dataset.replace('.json', '_direct_fourierft.json')
    if fourierft_weights:
        adapter_name = os.path.basename(fourierft_weights)
        output_file = output_file.replace('_direct_fourierft.json', f'_{adapter_name}_direct_fourierft.json')
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    fire.Fire(evaluate_fourierft_direct)
