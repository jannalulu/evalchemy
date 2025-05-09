#!/usr/bin/env python3
"""
Simple Qwen3 Model Interaction Script
Edit the model_name and question variables directly in this script.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== EDIT THESE SETTINGS =====
# Choose your model: "qwen3-7b-instruct", "qwen3-72b-instruct", "qwen3-1.8b-instruct", etc.
model_name = "qwen3-4b"

# Set your system prompt (can be empty)
system_prompt = "" #You are a helpful AI assistant.

# Choose mode: "interactive" to chat continuously, or "single" to ask just one question
mode = "single"  # or "single"

# If using single mode, set your question here
question = "How are you?"
# ===============================

# Model paths lookup
MODEL_PATHS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Base",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B-Base",
    "qwen3-8b-instruct": "Qwen/Qwen3-8B",
    "qwen3-8b-instruct-base": "recursal/RWKV7Qwen3-8B-InstructBase-250504"
}

def main():
    # Check if model is valid
    if model_name not in MODEL_PATHS:
        print(f"Error: Model '{model_name}' not found. Available models are:")
        for model in MODEL_PATHS:
            print(f"  - {model}")
        return
    
    model_path = MODEL_PATHS[model_name]
    print(f"Loading {model_name} from {model_path}...")
    
    # Determine device (CUDA if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    
    # Run in selected mode
    if mode == "interactive":
        run_interactive_mode(model, tokenizer)
    else:
        response = get_response(model, tokenizer, system_prompt, question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {response}")

def get_response(model, tokenizer, system_prompt, user_question):
    """Get a response from the model for a single question"""
    if system_prompt:
        input_text = f"System: {system_prompt}\n\n <|im_start|>User: {user_question}<|im_end|>\n\nAssistant:"
    else:
        input_text = f"<|im_start|>User\n{user_question}<|im_end|>\n<|im_start|>Assistant"
    
    # Encode the input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full_response[len(input_text):].strip()

def run_interactive_mode(model, tokenizer):
    """Run an interactive chat session with the model"""
    print("\n" + "="*50)
    print("Interactive Mode - Type 'exit' or 'quit' to end")
    print("="*50 + "\n")
    
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            response = get_response(model, tokenizer, system_prompt, user_input)
            print(f"\nAssistant: {response}")
    
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()