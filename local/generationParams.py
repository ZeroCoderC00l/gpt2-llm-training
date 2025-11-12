import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load model and tokenizer
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Fix padding token issue
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()


def generate_text(prompt, method="balanced"):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        if method == "conservative":
            # Most controlled
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                temperature=0.5,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )
        elif method == "greedy":
            # Most predictable
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elif method == "beam":
            # Highest quality
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
        else:  # balanced
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Test different methods
prompt = "Here's a programming joke about Java:\nQ: Why do Java developers wear glasses?\nA:"

print("=== BALANCED ===")
print(generate_text(prompt, "balanced"))
print("\n=== CONSERVATIVE ===")
print(generate_text(prompt, "conservative"))
print("\n=== GREEDY ===")
print(generate_text(prompt, "greedy"))
print("\n=== BEAM SEARCH ===")
print(generate_text(prompt, "beam"))