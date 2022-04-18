from tkinter import *
import tkinter.filedialog
from PIL import ImageTk, Image


class MlpMaker(tkinter.Tk):
    def __init__(self):
        self.App = Tk()
        self.App.title('MLP Designer')
        self.App.call('wm', 'iconphoto', self.App._w,
                      ImageTk.PhotoImage(Image.open('icons8-neural-network-64.ico')))
        self.App.geometry('1000x600')
        self.App['background'] = 'white'

        self.layers_label = Label(self.App, text='Layers')
        self.layers_label.grid(row=0, column=0, padx=25, pady=5)

        self.num_of_layers = Entry(self.App, background='white', width=10)
        self.num_of_layers.grid(row=0, column=1, padx=25, pady=5)

        self.labels = []
        self.input_spaces = []
        # self.inputs = []

        self.to_details = Button(self.App, text='Next',
                                 command=self.designer_details, relief='groove',
                                 borderwidth=4, state=NORMAL)

        self.to_last_details = Button(self.App, text='Next',
                                      command=self.last_details,
                                      relief='groove', borderwidth=4,
                                      state=NORMAL)
        # Activation function dropdown menu
        self.activ_label = Label(self.App, text='Activation Function', width=15)
        self.activ_func = StringVar()
        self.activ_menu = OptionMenu(self.App, self.activ_func, 'Sigmoid',
                                     'Tanh', 'ReLU')
        # Dropout percentage
        self.dropout_label = Label(self.App, text='Dropout(%)', width=15)
        self.dropout_entry = Entry(self.App, background='white', width=10)
        # Optimizer dropdown menu
        self.optim_label = Label(self.App, text='Optimizer', width=15)
        self.optim = StringVar()
        self.optim_menu = OptionMenu(self.App, self.optim, 'Adam', 'SGD')
        # Loss function dropdown menu
        self.loss_f_label = Label(self.App, text='Loss Function', width=15)
        self.loss_f = StringVar()
        self.loss_f_menu = OptionMenu(self.App, self.loss_f, 'MSE',
                                      'Cross Entropy Loss',
                                      'Binary Cross Entropy Loss')
        # Learning Rate
        self.lr_label = Label(self.App, text='Learning Rate', width=15)
        self.lr_entry = Entry(self.App, background='white', width=10)
        # Epochs
        self.ep_label = Label(self.App, text='Epochs', width=15)
        self.ep_entry = Entry(self.App, background='white', width=10)
        # Training Device
        self.tr_device_label = Label(self.App, text='Training Device', width=15)
        self.tr_device = StringVar()
        self.tr_device_chk_gpu = Checkbutton(self.App, variable=self.tr_device,
                                             text='GPU', onvalue='gpu',
                                             offvalue='')
        self.tr_device_chk_cpu = Checkbutton(self.App, variable=self.tr_device,
                                             text='CPU', onvalue='cpu',
                                             offvalue='')
        self.tr_device_chk_gpu.deselect()
        self.tr_device_chk_cpu.deselect()
        # Task at hand
        self.task_label = Label(self.App, text='Task', width=15)
        self.task = StringVar()
        self.task_menu = OptionMenu(self.App, self.task, 'Regression',
                                    'Binary Classification',
                                    'Classification')
        # Dataset Entry
        self.dataset_label = Label(self.App, text='Train Dataset', width=15)
        self.dataset_entry = Entry(self.App, background='white', width=80)
        self.dataset_choose = Button(self.App, text='Browse',
                                      command=self.get_directory,
                                      relief='groove', borderwidth=4)

    def layer_maker(self):
        layers = int(self.num_of_layers.get())

        self.labels.append(Label(self.App, text='Input Layer', width=10))
        self.labels[0].grid(row=1, column=0, padx=25, pady=5)

        self.input_spaces.append(Entry(self.App, background='white', width=10))
        self.input_spaces[0].grid(row=1, column=1, padx=25, pady=5)

        for i in range(1, layers - 1):
            txt = 'Layer ' + str(i)
            self.labels.append(Label(self.App, text=txt, width=10))
            self.labels[i].grid(row=i + 1, column=0, padx=25, pady=5)

            self.input_spaces.append(Entry(self.App, background='white', width=10))
            self.input_spaces[i].grid(row=i + 1, column=1, padx=25, pady=5)

        self.labels.append(Label(self.App, text='Last Layer', width=10))
        self.labels[layers - 1].grid(row=layers, column=0, padx=25, pady=5)

        self.input_spaces.append(Entry(self.App, background='white', width=10))
        self.input_spaces[layers - 1].grid(row=layers, column=1, padx=25, pady=5)

        self.to_details.grid(row=layers, column=3, padx=25, pady=5)

    def designer_details(self):
        self.to_details['state'] = DISABLED
        # Activation function dropdown menu
        self.activ_label.grid(row=1, column=4, padx=25, pady=5)
        self.activ_menu.grid(row=1, column=5, padx=25, pady=5)
        # Dropout percentage
        self.dropout_label.grid(row=2, column=4, padx=25, pady=5)
        self.dropout_entry.grid(row=2, column=5, padx=25, pady=5)
        # Optimizer
        self.optim_label.grid(row=3, column=4, padx=25, pady=5)
        self.optim_menu.grid(row=3, column=5, padx=25, pady=5)
        # Loss Function
        self.loss_f_label.grid(row=4, column=4, padx=25, pady=5)
        self.loss_f_menu.grid(row=4, column=5, padx=25, pady=5)
        # Learning Rate
        self.lr_label.grid(row=5, column=4, padx=25, pady=5)
        self.lr_entry.grid(row=5, column=5, padx=25, pady=5)
        # Epochs
        self.ep_label.grid(row=6, column=4, padx=25, pady=5)
        self.ep_entry.grid(row=6, column=5, padx=25, pady=5)

        self.to_last_details.grid(row=6, column=6, padx=25, pady=5)

    def last_details(self):
        self.to_last_details['state'] = DISABLED
        r = int(self.num_of_layers.get())
        # Training Device
        self.tr_device_label.grid(row=7, column=4, padx=25, pady=5)
        self.tr_device_chk_cpu.grid(row=7, column=5, padx=25, pady=5)
        self.tr_device_chk_gpu.grid(row=7, column=6, padx=25, pady=5)
        # Task
        self.task_label.grid(row=8, column=4, padx=25, pady=5)
        self.task_menu.grid(row=8, column=5, padx=25, pady=5)
        # Dataset entry
        self.dataset_label.grid(row=r+5, column=0, padx=25, pady=5)
        self.dataset_entry.grid(row=r+5, column=1, columnspan=5, padx=25, pady=5)
        self.dataset_choose.grid(row=r+5, column=6, padx=25, pady=5)

    def get_directory(self):
        dataset_dir = tkinter.filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        self.dataset_entry.insert(index=0, string=dataset_dir)

    def get_mlp_init_values(self):
        mlp_values = dict()
        mlp_values['num_of_layers'] = int(self.num_of_layers.get())
        layer_sizes = []
        try:
            for layer_size in self.input_spaces:
                layer_sizes.append(int(layer_size.get()))
        except ValueError:
            print('Error')
        mlp_values['layer_sizes'] = layer_sizes
        mlp_values['activ_f'] = self.activ_func.get()
        mlp_values['dropout'] = int(self.dropout_entry.get())
        mlp_values['optim'] = self.optim.get()
        mlp_values['loss_f'] = self.loss_f.get()
        mlp_values['lr'] = float(self.lr_entry.get())
        mlp_values['device'] = self.tr_device.get()
        mlp_values['epochs'] = int(self.ep_entry.get())
        mlp_values['task'] = self.task.get()
        mlp_values['dataset_path'] = self.dataset_entry.get()

        print(mlp_values)
        return mlp_values

    def refresh_window(self):
        # Initialize and delete everything again
        self.to_details['state'] = NORMAL
        self.to_last_details['state'] = NORMAL

        self.num_of_layers.delete(0, 'end')
        for entry in self.input_spaces:
            entry.destroy()
        for label in self.labels:
            label.grid_remove()

        self.to_details.grid_remove()
        self.to_last_details.grid_remove()
        self.labels = []
        self.input_spaces = []

        # Activation function dropdown menu
        self.activ_label.grid_remove()
        self.activ_menu.grid_remove()
        self.activ_func.set('')
        # Dropout
        self.dropout_label.grid_remove()
        self.dropout_entry.delete(0, 'end')
        self.dropout_entry.grid_remove()
        # Optimizer
        self.optim_label.grid_remove()
        self.optim_menu.grid_remove()
        self.optim.set('')
        # Loss Function
        self.loss_f_label.grid_remove()
        self.loss_f_menu.grid_remove()
        self.loss_f.set('')
        # Learning Rate
        self.lr_label.grid_remove()
        self.lr_entry.delete(0, 'end')
        self.lr_entry.grid_remove()
        # Epochs
        self.ep_label.grid_remove()
        self.ep_entry.delete(0, 'end')
        self.ep_entry.grid_remove()
        # Training Device
        self.tr_device_label.grid_remove()
        self.tr_device.set('')
        self.tr_device_chk_cpu.grid_remove()
        self.tr_device_chk_gpu.grid_remove()
        # Dataset entry
        self.dataset_label.grid_remove()
        self.dataset_entry.delete(0, 'end')
        self.dataset_entry.grid_remove()
        self.dataset_choose.grid_remove()
