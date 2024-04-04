import sys; sys.path.append('./')

import re, collections, os, json
from src.tokenizer import Tokenizer
from tqdm import tqdm

class BytePairTokenizer(Tokenizer):
    def __init__(self, name="byte_pair", *args, **kwargs):
        super(BytePairTokenizer, self).__init__(name=name, *args, **kwargs)
        self.name = name
    
    def pre_tokenize(self, text):
        text = self.normalize(text)
        return re.findall("\w+|[^a-z\s]", text)
    

    def build(self, corpus):
        word_freqs = self.pre_build(corpus)
        splits = {word: [char for char in word] for word in word_freqs.keys()}
        self.vocab = list(set([char for word in word_freqs.keys() for char in word]))
        self.merges = {}
        progress_bar = tqdm(total=self.max_tokens,desc="Build %s tokenizer ..." % self.name)
        while len(self.vocab) < self.max_tokens:
            progress_bar.update(1)
            try:
                pair_freqs = self.computes_pair_freqs(word_freqs, splits)
                best_pair, _ = self.find_best_pair(pair_freqs)
                splits = self.merge_pair(*best_pair, splits, word_freqs)
                self.merges[best_pair] = best_pair[0]+best_pair[1]
                self.vocab.append(best_pair[0]+best_pair[1])
            except: break
        
        self.id_to_token = [self.pad_token] + sorted(set(
        self.special_tokens + self.vocab
        ))
        self.token_to_id = {token: idx
                            for idx, token in enumerate(self.id_to_token)}   

    
    def computes_pair_freqs(self, word_freqs, splits):
        pair_freqs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split)==1: continue
            for i in range(len(split)-1):
                pair = (split[i],split[i+1])
                pair_freqs[pair]+=freq
        return pair_freqs

    def find_best_pair(self, pair_freqs):
        return max(pair_freqs.items(), key=lambda x:x[1])
    
    def merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            if len(split)==1: continue
            i = 0
            while i < len(split)-1:
                if split[i]==a and split[i+1]==b:
                    split = split[:i] + [a+b] + split[i+2:]
                else: i+=1
            splits[word] = split
        return splits
    
    def tokenize(self, text):
        if isinstance(text, list):
            return [self.tokenize(seq) for seq in text]
        pre_tokenized_text = self.pre_tokenize(text)
        
        splits = [[char for char in word] for word in pre_tokenized_text]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split)-1:
                    if split[i]==pair[0] and split[i+1]==pair[1]:
                        split = split[:i] + [merge] + split[i+2:]
                    else: i+=1
                splits[idx] = split
        return sum(splits, [])
    
    def get_config(self):
        config = super().get_config()
        config["merges"] = {'|'.join(k):v for k,v in self.merges.items()}
        return config

    @classmethod
    def load(cls, folder):
        config_filename = os.path.join(folder, "config.json")
        with open(config_filename, "r") as config_file:
            config = json.load(config_file)
        tokenizer = cls(config)

        tokenizer.special_tokens = config["special_tokens"]
        tokenizer.vocab = config["vocab"]
        tokenizer.id_to_token = config["id_to_token"]
        tokenizer.token_to_id = config["token_to_id"]
        tokenizer.merges = {tuple(k.split('|')): v for k,v in config["merges"].items()}
        return tokenizer
