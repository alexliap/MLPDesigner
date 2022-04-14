import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def get_dataset(df):
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

