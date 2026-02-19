# Copyright (c) Microsoft. All rights reserved.

"""
python fsdp_to_hf.py --fsdp-model-path checkpoints/AgentLightning/spider/global_step_192/actor/model_world_size_1_rank_0.pt --hf-tokenizer-path checkpoints/AgentLightning/spid
er/global_step_192/actor/huggingface --output-dir checkpoints/AgentLightning/spider/global_step_192/hf_model
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HuggingFace format.")
    parser.add_argument(
        "--fsdp-model-path",
        default="checkpoints/AgentLightning/spider/global_step_32/actor/model_world_size_1_rank_0.pt",
        help="Path to the FSDP state dict .pt file.",
    )
    parser.add_argument(
        "--hf-model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HuggingFace model name or path for the base architecture.",
    )
    parser.add_argument(
        "--hf-tokenizer-path",
        default="checkpoints/AgentLightning/spider/global_step_32/actor/huggingface",
        help="Path to the HuggingFace tokenizer.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/AgentLightning/spider/global_step_32/hf_model",
        help="Directory to save the converted HuggingFace model.",
    )
    args = parser.parse_args()

    # Load the FSDP state dict
    state_dict = torch.load(args.fsdp_model_path, map_location="cpu")

    # Load the base model architecture
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16)

    # Load trained weights into the model
    model.load_state_dict(state_dict)

    # Save in HuggingFace format
    model.save_pretrained(args.output_dir)

    # Copy tokenizer files too
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_path)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()