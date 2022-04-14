from tkinter import *
from mlp_maker import MlpMaker
from network import MlpNetwok
from pytorch_lightning import Trainer
import pandas as pd
import data_processing


def train():
    parameters = app.get_mlp_init_values()
    neuroniko = MlpNetwok(parameters['num_of_layers'],
                          parameters['layer_sizes'],
                          parameters['activ_f'], parameters['dropout'],
                          parameters['optim'], parameters['loss_f'],
                          parameters['lr']).double()
    if parameters['device'] == 'gpu':
        trainer = Trainer(gpus=1, max_epochs=parameters['epochs'])
    else:
        trainer = Trainer(gpus=0, max_epochs=parameters['epochs'])
    train_dataset = pd.read_csv(parameters['dataset_path']).drop('Unnamed: 0',
                                                                 axis=1)
    train_loader, val_loader = data_processing.get_dataset(train_dataset)

    trainer.fit(neuroniko, train_loader, val_loader)


if __name__ == '__main__':
    app = MlpMaker()

    refresh_button = Button(app.App, text="Reset", command=app.refresh_window,
                            relief='groove', borderwidth=4)
    refresh_button.place(x=700, y=500, relheight=0.08, relwidth=0.08)

    button1 = Button(app.App, text='Next', command=app.layer_maker,
                     relief='groove',
                     borderwidth=4)
    button1.grid(row=0, column=3, padx=25, pady=5)

    # button2 = Button(app.App, text='Get', command=app.get_mlp_init_values,
    #                  relief='groove', borderwidth=4)
    # button2.place(x=100, y=500, relheight=0.08, relwidth=0.08)

    train_b = Button(app.App, text='Train', command=train, relief='groove',
                     borderwidth=4, background='green')
    train_b.place(x=600, y=500, relheight=0.08, relwidth=0.08)

    app.App.mainloop()
