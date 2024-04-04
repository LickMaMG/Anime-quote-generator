import sys; sys.path.append('.')

from collections import defaultdict
import torch, time
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src.parameters import Parameters

class Trainer(Parameters):
    def __init__(self,
                 max_epochs, model_name="transformer", gradient_clip_value=0):
        super().__init__()
        self.save_parameters(max_epochs=max_epochs,
                             gradient_clip_value=gradient_clip_value)
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available()
                       else torch.device("cpu"))
        
        log_dir = time.strftime("logs/{}--%Y-%m-%d-%H-%M-%S".format(model_name))
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def prepare_batch(self, batch):
        return [tensor.to(self.device) for tensor in batch]
    
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        
        self.num_train_batch = len(self.train_dataloader)
        self.num_val_batch = (len(self.val_dataloader)
                              if self.val_dataloader is not None else 0)
        
    def prepare_model(self, model):
        model.trainer = self
        self.model = model.to(self.device)
    
    def fit(self, model, data):
        self.prepare_model(model)
        self.prepare_data(data)
        self.optimizer = model.configure_optimizers()

        self.epoch = 0
        for self.epoch in range(self.max_epochs): self.fit_epoch()

        self.writer.close()

    def fit_epoch(self):
        self.model.train()
        print("Epoch %d/%d" % (
            self.epoch+1, self.max_epochs))
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            batch = self.prepare_batch(batch)
            loss, metrics = self.model.training_step(batch)
            loss.backward()
            if self.gradient_clip_value > 0:
                self.clip_gradients(self.gradient_clip_value, self.model)
            self.optimizer.step()

            self.history(batch_idx=batch_idx, loss=loss, metrics=metrics)
            
        if self.val_dataloader is None: return
        self.model.eval()
        
        val_loss = 0.
        val_metrics = defaultdict(int)
        
        for batch in self.val_dataloader:
            with torch.no_grad():
                batch = self.prepare_batch(batch)
                val_loss_step, val_metrics_step = self.model.validation_step(batch)
                val_loss += val_loss_step

                for name, value in val_metrics_step.items():
                    val_metrics[name] += value
                

        val_loss /= self.num_val_batch
        for name in val_metrics: val_metrics[name] /= self.num_val_batch
        
        self.val_history(val_loss=val_loss, val_metrics=val_metrics)
        

    def clip_gradients(self, gradient_clip_value, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > gradient_clip_value:
            for param in params:
                param.grad[:] *= gradient_clip_value / norm
    

    def history(self, batch_idx, loss, metrics):
        progress = (batch_idx+1) / self.num_train_batch
        bar_length = 30
        complete_length = int(bar_length*progress)
        bar = '[' + '='*complete_length + '>' + '.'*(bar_length-complete_length-1) + ']'
        if complete_length==bar_length:
            bar = '[' + '='*complete_length + ']'
        history = "\r%d/%d %s - loss: %.4f" % (
                batch_idx+1, self.num_train_batch, bar, loss)
        
        for name, value in metrics.items():
            history += " - %s: %.4f" % (name, value)
            self.writer.add_scalar("training %s" % name, value.item(), self.epoch+1)
        
        self.writer.add_scalar("training loss", loss.item(), self.epoch+1)

        print(history, end='')


    def val_history(self, val_loss, val_metrics):
        val_history = " - val_loss: %.4f" % val_loss
        for name, value in val_metrics.items():
            val_history += " - val_%s: %.4f" % (name, value)
            self.writer.add_scalar("val %s" % name, value.item(), self.epoch+1)

        self.writer.add_scalar("val loss", val_loss.item(), self.epoch+1)
        print(val_history)
    