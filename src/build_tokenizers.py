import sys; sys.path.append('.')
from src.byte_pair_tokenizer import BytePairTokenizer
from src.tokenizer import Tokenizer
import pandas as pd

MAX_TOKENS = 10000

def main():
    df1 = pd.read_csv("data/AnimeQuotes.csv", usecols=["Quote"])
    df2 = pd.read_parquet("data/train-00000-of-00001.parquet", columns=["Quote"])
    df2.Quote = df2.Quote.map(lambda x:' '.join(x.split(": ")[1:]))
    df = pd.concat([df1,df2])
    df.Quote = df.Quote.map(lambda x:x.strip())
    quotes = df.Quote.values.tolist()
    quotes = [q for q in quotes if len(q)>0]
    # df = pd.read_csv("data/AnimeQuotes.csv", usecols=["Quote"])

    # name = "tokenizer"
    # tokenizer = Tokenizer(
    name = "byte-pair-tokenizer"
    tokenizer = BytePairTokenizer(
        name=name,
        max_tokens=MAX_TOKENS,
        min_freq=2
    )
    tokenizer.build(corpus=quotes)
    tokenizer.save(name)
    print("vocab_size:", tokenizer.num_tokens)

if __name__ == "__main__": main()