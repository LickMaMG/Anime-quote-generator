import sys; sys.path.append('.')
import os, torch

from src.data_module import AnimeQuoteModule
from src.tokenizer import Tokenizer
from src.model import Transformer
from src.trainer import Trainer
from src.data_module import AnimeQuoteModule

tokenizer = Tokenizer.load("tokenizer")

translation_data_module = AnimeQuoteModule(
    train_filename="data/quotes-data-train.txt", val_filename="data/quotes-data-val.txt",
    tokenizer=tokenizer, batch_size=128, num_steps=33
)

num_hiddens = 512
num_blocks = 2
dropout = 0.2
ffn_depth = 64
num_heads = 4

transformer = Transformer(
    num_blocks=num_blocks, num_hiddens=num_hiddens,
    num_heads=num_heads, ffn_depth=ffn_depth,
    target_vocab_size=tokenizer.vocab_size,
    dropout=dropout,
    lr=0.001)

trainer = Trainer(max_epochs=100, gradient_clip_value=1)
print("Train on", trainer.device)
trainer.fit(transformer, translation_data_module)
os.makedirs("./models", exist_ok=True)
# torch.save({
#     "model_state_dict": trainer.model.state_dict(),
#     # "optimizer_state_dict": trainer.optimizer.state_dict()
#     },
#     "./models/pt-transformer/pt-transformer.pth")
torch.save(trainer.model, "./models/quotes-generator.pth")
