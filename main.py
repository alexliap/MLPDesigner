import tkinter
from mlp_maker import MlpMaker
from network import MlpNetwork
from pytorch_lightning import Trainer
import pandas as pd
import data_processing


def train():
    parameters = app.get_mlp_init_values()
    neuroniko = MlpNetwork(parameters['num_of_layers'],
                           parameters['layer_sizes'],
                           parameters['activ_f'], parameters['dropout'],
                           parameters['optim'], parameters['loss_f'],
                           parameters['lr'], parameters['task'])

    if parameters['device'] == 'gpu':
        trainer = Trainer(gpus=1, max_epochs=parameters['epochs'])
    else:
        trainer = Trainer(gpus=0, max_epochs=parameters['epochs'])
    train_dataset = pd.read_csv(parameters['dataset_path'])

    if 'Unnamed: 0' in train_dataset.columns:
        train_dataset.drop('Unnamed: 0', axis=1, inplace=True)

    train_dataset = train_dataset.select_dtypes(exclude='object')
    train_loader, val_loader = data_processing.get_dataset(train_dataset)

    trainer.fit(neuroniko, train_loader, val_loader)

    train_status = tkinter.Label(app.App, text='Training Status: Done', width=20)
    train_status.grid(row=neuroniko.num_of_layers+6, column=4, padx=25,
                      pady=5)

    if parameters['task'] != 'Regression':
        lines1 = data_processing.train_loss_graphs(neuroniko.train_loss_values,
                                                   neuroniko.val_loss_values)
        lines2 = data_processing.train_acc_scores(neuroniko.train_acc,
                                                  neuroniko.val_acc)
        data_processing.merge_graphs(lines1, lines2)
    else:
        data_processing.train_loss_graphs(neuroniko.train_loss_values,
                                          neuroniko.val_loss_values,
                                          show=True)


# def predict():
#     model = MlpNetwok.load_from_checkpoint('lightning_logs/version_19/checkpoints/epoch=14-step=105.ckpt')
#     trainer = Trainer()
#     trainer.predict(model, )
#     pass


if __name__ == '__main__':
    app = MlpMaker('gray')

    refresh_button = tkinter.Button(app.App, text="Reset",
                                    command=app.refresh_window, relief='groove',
                                    borderwidth=4)
    refresh_button.place(x=700, y=500, relheight=0.08, relwidth=0.08)

    button1 = tkinter.Button(app.App, text='Next', command=app.layer_maker,
                             relief='groove', borderwidth=4)
    button1.grid(row=0, column=3, padx=25, pady=5)

    train_b = tkinter.Button(app.App, text='Train', command=train,
                             relief='groove', borderwidth=4, background='green')
    train_b.place(x=500, y=500, relheight=0.08, relwidth=0.08)

    predict_b = tkinter.Button(app.App, text='Predict', command=train,
                               relief='groove', borderwidth=4, background='red')
    predict_b.place(x=600, y=500, relheight=0.08, relwidth=0.08)

    app.App.mainloop()
