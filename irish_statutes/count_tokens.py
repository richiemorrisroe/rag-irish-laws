import argparse

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--file")

args = parser.parse_args()

print(args)

if args.file:
    print(args.file)
    with open(args.file, 'r') as f:
        text = f.read()
else:
    text = "Write your text here"

print(text)    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

tokens = tokenizer.tokenize(text)
num_tokens = len(tokens)
print(f"Number of tokens in your text: {num_tokens}")
