import sys; sys.path.append('.')
import torch
from src.tokenizer import Tokenizer
from src.byte_pair_tokenizer import BytePairTokenizer
from src.generator import Generator

model = torch.load("./models/quotes-generator.pth")
# tokenizer = Tokenizer.load("tokenizer")
tokenizer = BytePairTokenizer.load("byte-pair-tokenizer")

generator = Generator(tokenizer=tokenizer, model=model, temperature=0.8)
quotes = ["People's", "If you", "Being week", "Why"]
for q in quotes:
    text = generator(q)
    print(text)
    
