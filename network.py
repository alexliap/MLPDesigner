import torch
from pytorch_lightning import LightningModule
from torch.nn import ModuleList, Linear, ReLU, Sigmoid, Tanh, Dropout, MSELoss, \
                     BCELoss, CrossEntropyLoss, BatchNorm1d
from torch.optim import Adam, SGD
import torch.nn.functional as F


class MlpNetwok(LightningModule):
    def __init__(self, num_of_layers, layer_sizes, activ_f, dropout, optim,
                 loss_f, lr, task):
        super(MlpNetwok, self).__init__()
        # Activation Function
        if activ_f == 'ReLU':
            self.activ_f = ReLU()
        elif activ_f == 'Sigmoid':
            self.activ_f = Sigmoid()
        else:
            self.activ_f = Tanh()

        if loss_f == 'MSE':
            self.loss_f = MSELoss()
        elif activ_f == 'Cross Entropy Loss':
            self.loss_f = CrossEntropyLoss()
        else:
            self.loss_f = BCELoss()

        self.epoch = 0
        self.num_of_layers = num_of_layers
        self.optim = optim
        self.dropout = Dropout(dropout/100)
        self.lr = lr
        self.task = task
        self.layer_list = ModuleList()
        self.batch_norm_list = ModuleList()
        for i in range(self.num_of_layers-1):
            self.layer_list.append(Linear(layer_sizes[i], layer_sizes[i+1]))
        for j in range(self.num_of_layers-1):
            self.batch_norm_list.append(BatchNorm1d(layer_sizes[j]))

        self.train_loss_values = []
        self.train_acc = []

        self.val_loss_values = []
        self.val_acc = []

        self.predictions = []

    def forward(self, x):
        for i in range(self.num_of_layers-2):
            x = self.batch_norm_list[i](x)
            x = self.layer_list[i](x)
            x = self.dropout(F.relu(x))
        x = self.batch_norm_list[self.num_of_layers-2](x)
        x = self.layer_list[self.num_of_layers-2](x)
        out = self.activ_f(x)
        return out

    def configure_optimizers(self):
        if self.optim == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = SGD(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        out = self.forward(x)
        out = out.to(torch.float32)
        loss = self.loss_f(out, y)
        self.train_loss_values.append(loss)

        return {'loss': loss, 'predictions': torch.round(out), 'targets': y}

    def training_epoch_end(self, training_step_outputs):
        if self.task != 'Regression':
            all_preds = []
            length = 0
            for d in training_step_outputs:
                length = length + d['predictions'].size(0)
                all_preds.append(sum(d['predictions'] == d['targets']))
            acc = 100*sum(all_preds) / length
            self.train_acc.append(acc)
        else:
            pass

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        out = self.forward(x)
        out = out.to(torch.float32)
        loss = self.loss_f(out, y)
        self.val_loss_values.append(loss)

        return {'loss': loss, 'predictions': torch.round(out), 'targets': y}

    def validation_epoch_end(self, val_step_outputs):
        if self.task != 'Regression':
            all_preds = []
            length = 0
            for d in val_step_outputs:
                length = length + d['predictions'].size(0)
                all_preds.append(sum(d['predictions'] == d['targets']))
            acc = 100*sum(all_preds) / length
            self.val_acc.append(acc)
        else:
            pass

    # def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
    #     x = batch
    #     out = self.forward(x)
    #     out = torch.round(out)
    #     for pred in out:
    #         self.predictions.append(pred)
    #     return out
