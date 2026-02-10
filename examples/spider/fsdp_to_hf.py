import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the FSDP state dict
state_dict = torch.load(
    "checkpoints/AgentLightning/spider/global_step_32/actor/model_world_size_1_rank_0.pt",
    map_location="cpu",
)

# Load the base model architecture
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
)

# Load trained weights into the model
model.load_state_dict(state_dict)

# Save in HuggingFace format
output_dir = "checkpoints/AgentLightning/spider/global_step_32/hf_model"
model.save_pretrained(output_dir)

# Copy tokenizer files too
tokenizer = AutoTokenizer.from_pretrained(
    "checkpoints/AgentLightning/spider/global_step_32/actor/huggingface"
)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")