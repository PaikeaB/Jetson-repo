import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Example patterns for sensitive information
patterns = {
    'name': r'\b[A-Z][a-z]*\b',  # Simplistic pattern for names
    'location': r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b',  # Simplistic pattern for locations
    'password': r'(?i)\bpassword\b\s*:\s*\S+',  # Pattern for passwords
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',  # Pattern for credit card numbers
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Pattern for emails
    'phone': r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b',  # Pattern for phone numbers
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'birthdate': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'  # Pattern for birthdates
}

# Function to filter sensitive information
def filter_sensitive_info(text):
    for key, pattern in patterns.items():
        text = re.sub(pattern, f'[FILTERED_{key.upper()}]', text)
    return text

# Load pre-trained model and tokenizer
model_name = 'gpt2'  # You can replace this with the model you want to use
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(inputs.shape, device=device)  # Create an attention mask
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,  # Pass the attention mask
        max_length=max_length, 
        num_return_sequences=1,
        do_sample=True,  # Enable sampling
        temperature=temperature,  # Controls the randomness of predictions
        top_k=top_k,  # Limits the sampling pool
        top_p=top_p,  # Nucleus sampling
        pad_token_id=tokenizer.eos_token_id  # Ensures proper padding
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to generate and filter text
def generate_and_filter(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    generated_text = generate_text(prompt, max_length, temperature, top_k, top_p)
    filtered_text = filter_sensitive_info(generated_text)
    return filtered_text

# Main interaction loop
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt (or type 'exit' to stop): ")
        if prompt.lower() == 'exit':
            break
        filtered_response = generate_and_filter(prompt)
        print(f"Filtered Generated Text: {filtered_response}\n")

