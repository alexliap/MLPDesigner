import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk
import tkinter
from PIL import ImageTk, Image


def get_dataset(df: DataFrame):
    # We assume that the dataframe's last column are the target values
    # (or y values).
    x = df.iloc[:, :len(df.columns)-1].values
    y = df.iloc[:, len(df.columns)-1].values

    y = y.reshape(len(y), 1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.6,
                                                        random_state=50)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    train_dt = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dt, batch_size=2048)

    test_dt = TensorDataset(x_val, y_val)
    test_loader = DataLoader(test_dt, batch_size=2048)

    return train_loader, test_loader


def train_loss_plot(train_loss: list):
    loss = []
    for i in range(len(train_loss)):
        loss.append(train_loss[i].cpu().detach().numpy())

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(range(len(loss)), loss)
    ax.set_title('Train Loss')
    ax.set_xlabel('Steps')
    plt.show()

    train_window = tkinter.Toplevel()
    train_window.title('Loss Graphs')
    train_window.geometry('480x480')
    train_window['background'] = 'white'
    train_window.tk.call('wm', 'iconphoto', train_window._w,
                         ImageTk.PhotoImage(Image.open('icons8-neural-network-64.ico')))
    canvas = FigureCanvasTkAgg(fig, master=train_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, train_window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


