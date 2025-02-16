import os

import torch
from torch import nn
import pandas as pd

from torchvision.transforms import ToTensor, Compose, Normalize

print('Torch version:', torch.__version__)


class BaseMLPModel(nn.Module):
    def __init__(self, in_features, hid_features, n_classes):
        super(BaseMLPModel, self).__init__()
        
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, padding=0)
        )

        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(1024, hid_features)
        self.dropout = torch.nn.Dropout()
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hid_features, n_classes)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.flat(x)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu3(x)

        x = self.fc2(x)
        return x


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.ckpt')
        
        checkpoint = torch.load(model_path, weights_only=True)

        self.base_model = BaseMLPModel(**checkpoint['model_kwargs'])
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_f = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.base_model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.transform = Compose([
            # ToTensor(),
            Normalize([0.5], [0.5]),
        ])

        mapping = pd.read_csv('./emnist-balanced-mapping.txt', sep=' ', names=['label', 'code'])
        mapping['char'] = mapping['code'].apply(chr)
        mapping.set_index('label', inplace=True)
        self.__mapping = mapping

    def __lab2chr(self, label):
        return self.__mapping.loc[label]['char']     

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        if (x.shape == (28, 28)):
            x = x.reshape(1, 1, 28, 28)
            x = x.to(torch.float)
            x = self.transform(x)
            raw_pred = self.base_model.forward(x)[0]
            lab = raw_pred.argmax()
            ped_char = self.__lab2chr(int(lab))
            return ped_char
        else:
            raise ValueError(f'Shape of "x" should be (28, 28). Got: {x.shape}')

