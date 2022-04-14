from pytorch_lightning import LightningModule
from torch.nn import ModuleList, Linear, ReLU, Sigmoid, Tanh, Dropout, MSELoss, \
                    BCELoss, CrossEntropyLoss
from torch.optim import Adam, SGD
import torch.nn.functional as F
from tkinter import *
from tkinter import ttk


class MlpNetwok(LightningModule):
    def __init__(self, num_of_layers, layer_sizes, activ_f, dropout, optim,
                 loss_f, lr):
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
        # self.dev = device
        self.layer_list = ModuleList()
        for i in range(self.num_of_layers-1):
            self.layer_list.append(Linear(layer_sizes[i], layer_sizes[i+1]))

        self.train_loss_values = []
        self.val_loss_values = []
        # Progress Bar window
        # self.pb_window = Tk()
        # self.pb_window.title('Progress Bar')
        # self.pb_window.geometry('300x100')
        # self.pb_window['background'] = 'white'
        # self.pb = ttk.Progressbar(self.pb_window, orient='horizontal',
        #                           mode='indeterminate', length=280)
        # self.pb.pack()
        # self.epoch_label = Label(self.pb_window, text='Epoch: '+str(self.epoch))
        # self.epoch_label.pack(padx=25, pady=5)

    def forward(self, x):
        for i in range(self.num_of_layers-2):
            x = self.layer_list[i](x)
            x = self.dropout(F.relu(x))
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
        out = self.forward(x)
        loss = self.loss_f(out, y)
        # self.train_loss_values.append(loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.loss_f(out, y)
        # self.val_loss_values.append(loss)
        return loss

    # def training_epoch_end(self, outputs):
    #     self.pb.step()
