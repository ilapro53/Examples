import os
import pandas as pd
import numpy as np
import pickle


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')

        mapping = pd.read_csv('./emnist-balanced-mapping.txt', sep=' ', names=['label', 'code'])
        mapping['char'] = mapping['code'].apply(chr)
        mapping.set_index('label', inplace=True)
        self.__mapping = mapping

        with open(model_path, 'rb') as f:
            self.base_model = pickle.load(f)

        # self.__lab2chr_np = np.vectorize(self.__lab2chr)

    def __lab2chr(self, label):
        return self.__mapping.loc[label]['char']

    def predict(self, x: np.ndarray):
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
        if (x.shape == (28, 28)) or (x.shape == (784,)):
            raw_pred = self.base_model.predict(x.reshape(1, 28*28))[0]
            decoded_pred = self.__lab2chr(raw_pred)
            return decoded_pred
        else:
            raise ValueError(f'Shape of "x" should be (28, 28). Got: {x.shape}')
        


