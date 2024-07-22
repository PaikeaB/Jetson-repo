import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_blank(sentence, blank_token="_", top_k=5):
    # Split the sentence at the blank token
    parts = sentence.split(blank_token)
    if len(parts) != 2:
        raise ValueError("The sentence must contain exactly one blank token")
    
    prefix, suffix = parts
    
    # Tokenize the prefix
    prefix_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
    
    # Generate attention mask
    attention_mask = torch.ones(prefix_ids.shape, device=device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            prefix_ids,
            attention_mask=attention_mask,
            max_length=prefix_ids.size(1) + 10,  # Add some room for generation
            num_return_sequences=top_k,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Adjust temperature for randomness
            top_k=top_k,  # Top k sampling
            top_p=0.9,  # Top p (nucleus) sampling
            pad_token_id=tokenizer.eos_token_id  # Ensure proper padding
        )
    
    # Decode predictions and extract the word for the blank
    predictions = []
    for output in outputs:
        completion = tokenizer.decode(output, skip_special_tokens=True)
        # Find the first word after the prefix
        completion_words = completion[len(prefix):].strip().split()
        if completion_words:
            predicted_word = completion_words[0]
            predictions.append(predicted_word)
    
    # Filter out duplicates and return top-k predictions
    unique_predictions = list(dict.fromkeys(predictions))  # Preserve order and remove duplicates
    return unique_predictions[:top_k]

# Example usage
sentence = "Paikea is _ college student"
predicted_words = predict_blank(sentence)
print(f"Predicted words for the blank: {predicted_words}")

