import collections, json, os

class Tokenizer:

    @classmethod
    def add_method(cls, func):
        setattr(cls, func.__name__, func)
        return func
   
    def __init__(self, name: str = "simple_tokenizer", max_tokens: int = 5000,
                 max_char_length = 20, min_freq=2, special_tokens=[], max_sequence_length=8,
                 unk_token="<unk>", pad_token="<pad>", sos_token="<sos>", eos_token="<oes>",):
        self.name = name
        self.max_tokens = max_tokens
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._sos_token = sos_token
        self._eos_token = eos_token
       
        self._max_sequence_length = max_sequence_length
        self._max_char_length = max_char_length
        self._min_freq = min_freq
       
        self.special_tokens = special_tokens + [unk_token, sos_token, eos_token]
        
    
    @property
    def sos_token(self): return self._sos_token
   
    @property
    def sos_token_id(self): return self.token_to_id[self.sos_token]

    @property
    def eos_token(self): return self._eos_token

    @property
    def eos_token_id(self): return self.token_to_id[self.eos_token]
   
    @property
    def pad_token(self): return self._pad_token
   
    @property
    def pad_token_id(self): return 0

    @property
    def unk_token(self): return self._unk_token
   
    @property
    def unk_token_id(self): return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self): return len(self.id_to_token)
   
    @property
    def max_sequence_length(self): return self._max_sequence_length

    @max_sequence_length.setter
    def max_sequence_length(self, value): self._max_sequence_length = value
   
    @property
    def max_char_length(self): return self._max_char_length
   
    @property
    def min_freq(self): return self._min_freq
    
    @property
    def num_tokens(self): return len(self.id_to_token)

    def normalize(self, text):
        text = text.replace("\u202f", ' ').replace("\xa0", ' ')
        no_space = lambda char, prev_char: char in ",.!?" and prev_char != ' '
        out = [' '+char if i>0 and no_space(char, text[i-1]) else char
            for i, char in enumerate(text.lower())]
        out = ''.join(out)
        return out.lower()

    def pre_tokenize(self, text): return self.normalize(text).split(' ')


    def pre_build(self, corpus):
        if isinstance(corpus, list):
            tokens = [word for sub_corpus in corpus for word in self.pre_tokenize(sub_corpus)]
        else: tokens = self.pre_tokenize(corpus)
        tokens = [word for word in tokens if len(word)<=self.max_char_length]
        return collections.Counter(tokens)

    def build(self, corpus):
        counter = self.pre_build(corpus)
        self.vocab = [token for token, freq in counter.items() if freq>=self.min_freq][:self.max_tokens]

        self.id_to_token = [self.pad_token] + sorted(set(
            self.special_tokens + self.vocab
        ))
        self.token_to_id = {token: idx
                            for idx, token in enumerate(self.id_to_token)}   

    def add_special_tokens(self, tokens):
        self.special_tokens += tokens
        self.add_tokens(tokens, special_tokens=True)   

    def add_tokens(self, tokens, special_tokens=False):
        tokens = list(set([token for token in tokens if token not in self.id_to_token]))
        if special_tokens:
            tokens = ["<%s>" % token.lower() for token in tokens]
        self.id_to_token += tokens
        self.token_to_id = {token: idx
                                for idx, token in enumerate(self.id_to_token)}   
        if not special_tokens: self.vocab += tokens

    def pad(self, tokens, padding=True, max_length=None):
        assert not isinstance(tokens[0], list)
        if padding==True:
            max_length = self.max_sequence_length if max_length is None else max_length
            pad_size = max_length-len(tokens)
        elif padding==False: pad_size=0
        # elif padding=="longest": max_length=len(tokens)
        else: raise ValueError("This padding style does not exist for single instance")
        return tokens + [self.pad_token]*pad_size

    def pad_batch(self, batch, padding="max_length", max_length=None):
        if padding=="longest":
            max_length = max(len(seq) for seq in batch)
        elif padding=="max_length":
            max_length = self.max_sequence_length if max_length is None else max_length
        if padding in [True, "max_length", "longest"]: padding=True
        return [self.pad(seq, padding=padding, max_length=max_length) for seq in batch]

    def truncate(self, tokens, max_length):
        assert not isinstance(tokens[0], list)
        return (
            tokens[:max_length]
            if len(tokens)>max_length
            else tokens
        )

    def truncate_batch(self, batch, max_length):
        return [self.truncate(seq, max_length) for seq in batch]

    def truncate_pad(self, tokens, padding=True, truncation=True, max_length=None):
        max_length = self.max_sequence_length if max_length is None else max_length
        tokens = self.truncate(tokens, max_length) if truncation else tokens
        tokens = self.pad(tokens, padding, max_length)
        return tokens

    def truncate_pad_batch(self, batch, padding=True, truncation=True, max_length=None):
        max_length = self.max_sequence_length if max_length is None else max_length
        if truncation:
            batch = self.truncate_batch(batch, max_length=max_length)
        batch = self.pad_batch(batch, padding=padding, max_length=max_length)
        return batch

    def encode(self, tokens):
        if isinstance(tokens[0], list):
            return [self.encode(sub_tokens) for sub_tokens in tokens]
        return [self.token_to_id.get(token, self.unk_token_id)
                for token in tokens]

    def decode(self, ids):
        if isinstance(ids[0], list):
            return [self.decode(sub_ids) for sub_ids in ids]
        return [self.id_to_token[token_id] for token_id in ids]

    def tokenize(self, text):
        if isinstance(text, list):
            return [self.tokenize(seq) for seq in text]
        return self.pre_tokenize(text)

    def __call__(self, text, padding=True, truncation=True, max_length=None):
        tokens = self.tokenize(text)
        if isinstance(tokens[0], list):
            tokens = [[self.sos_token]+sub_tokens+[self.eos_token] for sub_tokens in tokens]
            tokens = self.truncate_pad_batch(tokens, padding=padding, truncation=truncation, max_length=max_length)
        else:
            tokens = [self.sos_token]+tokens+[self.eos_token]
            tokens = self.truncate_pad(tokens, padding=padding, truncation=truncation, max_length=max_length)
        output = self.encode(tokens)
        # if return_tensor and padding and truncation: output = torch.tensor(output)
        return output
    
    def get_config(self):
        return {
            "name": self.name,
            "max_tokens": self.max_tokens,
            "max_char_length": self.max_char_length,
            "min_freq": self.min_freq,
            "max_sequence_length": self.max_sequence_length,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "id_to_token": self.id_to_token,
            "token_to_id": self.token_to_id
        }
    
    def save(self, filename):
        os.makedirs(filename, exist_ok=True)
        config = self.get_config()
        config_filename = os.path.join(filename, "config.json")
        with open(config_filename, "w") as config_file:
            json.dump(config, config_file,indent=4)
    
    
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
        return tokenizer
