import argparse

from transformers import AutoTokenizer

def count_tokens(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    print(f"Number of tokens in your text: {num_tokens}")
    return num_tokens


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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
count_tokens(tokenizer, text)
