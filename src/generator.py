import sys; sys.path.append('.')
import torch
from src.parameters import Parameters
from src.tokenizer import Tokenizer
from src.byte_pair_tokenizer import BytePairTokenizer

model = torch.load("./models/quotes-generator.pth")
tokenizer = Tokenizer.load("tokenizer")
# tokenizer = BytePairTokenizer.load("byte-pair-tokenizer")

# print(model)

class Generator(Parameters):
    def __init__(self, model, tokenizer, temperature=0.0):
        self.save_parameters(model=model,
                             tokenizer=tokenizer,
                             temperature=temperature
                             )
        self.model.eval()
    
    def __call__(self, sentence, max_length=33):
        assert isinstance(sentence, str)

        sentence = self.tokenizer(sentence,padding=False)[:-1]
        decoder_in = torch.tensor(sentence).unsqueeze(0).to(self.model.trainer.device)
        output, attention_weights = [], []
        for _ in range(max_length):
            with torch.no_grad():
                predictions = self.model(decoder_in, None, None)
                predictions = predictions[:, -1, :]
                if self.temperature==0.0:
                    predicted_id = predictions.argmax(dim=-1).unsqueeze(1)
                else:
                    predictions /= self.temperature
                    probabilities = torch.softmax(predictions, dim=-1)
                    predicted_id = torch.multinomial(probabilities, num_samples=1)

                decoder_in = torch.cat((decoder_in, predicted_id), dim=-1)
                output.append(predicted_id.item())

                if predicted_id.item() == self.tokenizer.eos_token_id:
                    break

        text = self.tokenizer.decode(sentence + output)
        return text
