# Install the transformers library
# pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_study_guide(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

def main():
    print("Welcome to the Homework Study Guide Generator!")
    while True:
        user_input = input("Enter a topic or question for your study guide (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        study_guide = generate_study_guide(user_input)
        print("\nGenerated Study Guide:\n")
        print(study_guide)
        print("\n")

if __name__ == "__main__":
    main()