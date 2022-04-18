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

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4,
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


def train_loss_graphs(train_loss: list, val_loss: list, show=False):
    train_loss_values = []
    for i in range(len(train_loss)):
        train_loss_values.append(train_loss[i].cpu().detach().numpy())

    val_loss_values = []
    for i in range(len(val_loss)):
        val_loss_values.append(val_loss[i].cpu().detach().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2)
    line1 = ax1.plot(range(len(train_loss_values)), train_loss_values)
    ax1.set_title('Train Loss')
    ax1.set_xlabel('Steps')

    line2 = ax2.plot(range(len(val_loss_values)), val_loss_values)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Steps')

    fig.tight_layout()

    if show:
        put_em_on(fig)

    return line1, line2


def train_acc_scores(train_acc: list, val_acc: list, show=False):
    train_acc_values = []
    for i in range(len(train_acc)):
        train_acc_values.append(train_acc[i].cpu().detach().numpy())

    val_acc_values = []
    for i in range(len(val_acc)):
        val_acc_values.append(val_acc[i].cpu().detach().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2)
    line1 = ax1.plot(range(len(train_acc_values)), train_acc_values)
    ax1.set_title('Train Accuracy')
    ax1.set_xlabel('Steps')

    line2 = ax2.plot(range(len(val_acc_values)), val_acc_values)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Steps')

    fig.tight_layout()

    if show:
        put_em_on(fig)

    return line1, line2


def merge_graphs(loss_lines: tuple, acc_lines: tuple):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(loss_lines[0][0].get_data()[0], loss_lines[0][0].get_data()[1])
    axs[0, 1].plot(loss_lines[1][0].get_data()[0], loss_lines[1][0].get_data()[1])
    axs[1, 0].plot(acc_lines[0][0].get_data()[0], acc_lines[0][0].get_data()[1])
    axs[1, 1].plot(acc_lines[1][0].get_data()[0], acc_lines[1][0].get_data()[1])

    axs[0, 0].set_title('Training')
    axs[0, 0].set_ylabel('Train Loss')

    axs[0, 1].set_title('Validation')
    axs[0, 1].set_ylabel('Validation Loss')

    axs[1, 0].set_ylabel('Training Accuracy (%)')
    axs[1, 0].set_xlabel('Steps')

    axs[1, 1].set_ylabel('Validation Accuracy(%)')
    axs[1, 1].set_xlabel('Steps')

    fig.tight_layout()

    put_em_on(fig)


def put_em_on(fig: plt.Figure()):
    window = tkinter.Toplevel()
    window.title('Loss Graphs')
    window.geometry('640x480')
    window['background'] = 'white'
    window.tk.call('wm', 'iconphoto', window._w,
                   ImageTk.PhotoImage(Image.open('icons8-neural-network-64.ico')))
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)