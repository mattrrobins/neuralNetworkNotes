#! python3

import numpy as np

class ShuffleBatchData():
    def __init__(self, data, batch_size, seed=10101):
        """
        """
        self.data = data
        self.batch_size = batch_size
        self.seed = seed
        self.idx = np.arange(data['x'].shape[1])
        self.__N = data['x'].shape[1]

        np.random.seed(seed)
    
    def get_batches(self):
        """
        """
        np.random.shuffle(self.idx)
        x_shuffled = self.data['x'][:, self.idx]
        y_shuffled = self.data['y'][:, self.idx]

        B = int(np.ceil(self.__N / self.batch_size))

        batches = []
        for i in range(B):
            x_aux = x_shuffled[:, (self.batch_size * i):(self.batch_size * (i + 1))]
            y_aux = y_shuffled[:, (self.batch_size * i):(self.batch_size * (i + 1))]
            batches.append({'x' : x_aux, 'y' : y_aux})

        return batches


if __name__ == '__main__':
    from pprint import pprint as pp


    x = np.arange(1000).reshape(10, 100)
    y = np.arange(100).reshape(-1, 100)
    data = {'x' : x, 'y' : y}

    batch_size = 2 ** 5
    batching = ShuffleBatchData(data, batch_size=batch_size)
    for i in range(10):
        batches = batching.get_batches()
        for batch in batches:
            #pp(batch['x'])
            pp(batch['y'])


