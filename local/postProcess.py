import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

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


def clean_output(text, prompt):
    """Extract clean, single response from generated text"""
    # Remove original prompt
    response = text[len(prompt):].strip()

    # Split on double newlines (paragraph breaks)
    paragraphs = response.split('\n\n')
    if paragraphs:
        response = paragraphs[0]

    # Stop at these markers (common for multiple jokes/stories)
    stop_markers = [
        '\nQ:', '\nJoke:', '\nHere\'s another',
        '\n---', '\nNext:', '\nAnother', '\n\n'
    ]

    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0]

    # Limit sentences for very long outputs
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    if len(sentences) > 4:
        response = '. '.join(sentences[:4]) + '.'
    elif sentences:
        response = '. '.join(sentences) + '.'

    # Clean up extra whitespace
    response = re.sub(r'\s+', ' ', response).strip()

    return response


def generate_clean_text(prompt, max_length=80):
    """Generate text with controls to prevent rambling"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=15,
            temperature=0.7,
            top_k=50,
            top_p=0.92,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_output(generated, prompt)


# Test with different joke prompts
print("=== TEST 1: Q&A Format ===")
prompt1 = "Q: Why do Java developers wear glasses?\nA:"
print(f"Prompt: {prompt1}")
print(f"Response: {generate_clean_text(prompt1)}\n")

print("=== TEST 2: Completion Style ===")
prompt2 = "A Java programmer walks into a bar and says"
print(f"Prompt: {prompt2}")
print(f"Response: {generate_clean_text(prompt2, max_length=60)}\n")

print("=== TEST 3: Direct Instruction ===")
prompt3 = "Here is a short joke about Java programming:\n"
print(f"Prompt: {prompt3}")
print(f"Response: {generate_clean_text(prompt3, max_length=70)}\n")

# Interactive mode
print("\n" + "=" * 50)
print("Interactive Mode (type 'quit' to exit)")
print("=" * 50 + "\n")

while True:
    user_prompt = input("Enter your prompt: ")

    if user_prompt.lower() == 'quit':
        break

    result = generate_clean_text(user_prompt, max_length=100)
    print(f"\n{result}\n")
    print("-" * 50 + "\n")