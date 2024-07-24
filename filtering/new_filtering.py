import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Example patterns for sensitive information
patterns = {
    'name': r'\b[A-Z][a-z]*\b|\b[a-z]+\b',  # Pattern for names including lowercase
    'location': r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b',  # Simplistic pattern for locations
    'password': r'(?i)\bpassword\b\s*:\s*\S+',  # Pattern for passwords
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',  # Pattern for credit card numbers
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Pattern for emails
    'phone': r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b',  # Pattern for phone numbers
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'birthdate': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2}, \d{4})\b'  # Pattern for birthdates in various formats
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

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

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

# Load the dataset
dataset = load_dataset("ai4privacy/pii-masking-300k")

# Take a smaller subset of the dataset
small_train_dataset = dataset["train"].select(range(1000))  # Adjust number of samples
small_eval_dataset = dataset["validation"].select(range(200))  # Adjust number of samples

# Function to process dataset
def tokenize_function(examples):
    return tokenizer(examples['source_text'], padding="max_length", truncation=True, max_length=128)

# Tokenize the datasets
tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,  # Reduce the number of epochs
    per_device_train_batch_size=16,  # Increase the batch size
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    logging_steps=10,  # Add logging
    evaluation_strategy="steps",  # Evaluate during training
    eval_steps=50,  # Evaluate every 50 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
output_dir = "./trained_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Main interaction loop
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt (or type 'exit' to stop): ")
        if prompt.lower() == 'exit':
            break
        max_length = int(input("Enter max length: "))
        temperature = float(input("Enter temperature (0.0 - 1.0): "))
        top_k = int(input("Enter top_k: "))
        top_p = float(input("Enter top_p (0.0 - 1.0): "))
        filtered_response = generate_and_filter(prompt, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
        print(f"Filtered Generated Text: {filtered_response}\n")

