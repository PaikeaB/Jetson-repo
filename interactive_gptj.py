import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100)

    # Decode and return output text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

print("Welcome to the GPT-J interactive demo. Type 'exit' to quit.")
while True:
    input_text = input("You: ")
    if input_text.lower() == "exit":
        break
    response = generate_response(input_text)
    print(f"GPT-J: {response}")

