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

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                        random_state=50)
    # Scaling the data.
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    x_train = torch.from_numpy(x_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)

    x_val = torch.from_numpy(x_val).to(torch.float32)
    y_val = torch.from_numpy(y_val).to(torch.float32)

    train_dt = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dt, batch_size=2048)

    val_dt = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dt, batch_size=2048)

    return train_loader, val_loader


def train_loss_graphs(train_loss: list, val_loss: list, show=False):
    # Here the standalone graphs for the train and validation losses
    # are created. They are not shown on a different window by default
    # in case other graphs need to be created.
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
    # Here the standalone graphs for the train and validation accuracy
    # are created. They are not shown on a different window by default
    # in case other graphs need to be created.
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
    # The merging of all the graphs into one figure.
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(loss_lines[0][0].get_data()[0], loss_lines[0][0].get_data()[1])
    axs[0, 1].plot(loss_lines[1][0].get_data()[0], loss_lines[1][0].get_data()[1])
    axs[1, 0].plot(acc_lines[0][0].get_data()[0], acc_lines[0][0].get_data()[1])
    axs[1, 1].plot(acc_lines[1][0].get_data()[0], acc_lines[1][0].get_data()[1])

    axs[0, 0].set_title('Training')
    axs[0, 0].set_ylabel('Train Loss')
    axs[0, 0].set_xlabel('Steps')

    axs[0, 1].set_title('Validation')
    axs[0, 1].set_ylabel('Validation Loss')
    axs[0, 1].set_xlabel('Steps')

    axs[1, 0].set_ylabel('Training Accuracy (%)')
    axs[1, 0].set_xlabel('Epochs')

    axs[1, 1].set_ylabel('Validation Accuracy(%)')
    axs[1, 1].set_xlabel('Epochs')

    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    fig.tight_layout()

    put_em_on(fig)


def put_em_on(fig: plt.Figure()):
    # PLacement of graphs to a separate tkinter window.
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