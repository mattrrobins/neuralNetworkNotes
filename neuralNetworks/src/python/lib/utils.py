#! python3

import numpy as np
from sklearn.model_selection import train_test_split


class ProcessData():
    def __init__(self, x, y, dev_perc, test_perc):
        """
        Parameters:
        -----------
        x : array_like
            x.shape = (features, examples)
        y : array_like
            y.shape = (label, examples)
        dev_perc : float
        test_perc : float

        Returns:
        --------
        None
        """
        self.x = x.T
        self.y = y.T
        self.dev_perc = dev_perc
        self.test_perc = test_perc

        self.split()
        self.normalize()

    def split(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        percent = self.dev_perc + self.test_perc
        x_train, x_aux, y_train, y_aux = train_test_split(
            self.x, self.y, test_size=percent)
        self.train = {'x': x_train.T, 'y': y_train.T}
        new_percent = self.test_perc / percent
        x_dev, x_test, y_dev, y_test = train_test_split(
            x_aux, y_aux, test_size=new_percent)
        self.dev = {'x': x_dev.T, 'y': y_dev.T}
        self.test = {'x': x_test.T, 'y': y_test.T}

    def normalize(self, z=None, eps=1e-8):
        """
        Parameters:
        -----------
        z : array_like
            Default : None - For initialization
        eps : float
            Default 1e-8 - For stability

        Returns:
        z_scale : array_like
        """
        if z == None:
            x = self.train['x']
            self.mu = np.mean(x, axis=1, keepdims=True)
            self.var = np.var(x, axis=1, keepdims=True)
            self.theta = 1 / np.sqrt(self.var + eps)
            self.train['x'] = self.theta * (x - self.mu)
            self.dev['x'] = self.theta * (self.dev['x'] - self.mu)
            self.test['x'] = self.theta * (self.test['x'] - self.mu)
        else:
            z_scale = self.theta * (z - self.mu)
            return z_scale
