import torch
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms

# def train_dataset(n_samples = 200, batch_size = 1):
#     X_train = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#     # # Leaving only labels 0 and 1 
#     idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
#                     np.where(X_train.targets == 1)[0][:n_samples])
#     # idx = np.stack([np.where(X_train.targets == i)[0][:n_samples] for i in range(10)], axis=1)
#     # idx = idx.reshape(-1)

#     X_train.data = X_train.data[idx]
#     X_train.targets = X_train.targets[idx]

#     train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False, pin_memory=True)
#     return train_loader


# def test_dataset(n_samples = 200, batch_size = 1):
#     X_test = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#     idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
#                     np.where(X_test.targets == 1)[0][:n_samples])
#     # idx = np.stack([np.where(X_test.targets == i)[0][:n_samples] for i in range(10)], axis=1)
#     # idx = idx.reshape(-1)

#     X_test.data = X_test.data[idx]
#     X_test.targets = X_test.targets[idx]

#     test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
#     return test_loader


def train_dataset(n_samples = 200, batch_size = 1):
    X_train = FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # # Leaving only labels 0 and 1 
    idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                    np.where(X_train.targets == 1)[0][:n_samples])
    # idx = np.stack([np.where(X_train.targets == i)[0][:n_samples] for i in range(10)], axis=1)
    # idx = idx.reshape(-1)

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader


def test_dataset(n_samples = 200, batch_size = 1):
    X_test = FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                    np.where(X_test.targets == 1)[0][:n_samples])
    # idx = np.stack([np.where(X_test.targets == i)[0][:n_samples] for i in range(10)], axis=1)
    # idx = idx.reshape(-1)

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
    return test_loader
