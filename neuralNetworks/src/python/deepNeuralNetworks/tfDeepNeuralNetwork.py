#! python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Model, Input
from keras.layers import Dense

def keras_functional_nn(csv):
    df = pd.read_csv(csv)
    dataset = df.values
    x, y = dataset[:, :-1], dataset[:, -1].reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    train = {'x' : x_train, 'y' : y_train}
    test = {'x' : x_test, 'y' : y_test}
    mu = np.mean(train['x'], axis=0, keepdims=True)
    var = np.var(train['x'], axis=0, keepdims=True)
    train['x'] = (train['x'] - mu) / np.sqrt(var)
    test['x'] = (test['x'] - mu) / np.sqrt(var)

    ## Define network structure
    input_layer = Input(shape=(10,))
    hidden_layer_1 = Dense(
        32,
        activation='relu',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )(input_layer)
    hidden_layer_2 = Dense(
        8,
        activation='relu',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )(hidden_layer_1)
    output_layer = Dense(
        1,
        activation='sigmoid',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )(hidden_layer_2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    ## Compile desired model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    ## Train the model
    hist = model.fit(
        train['x'],
        train['y'],
        batch_size=32,
        epochs=150,
        validation_split=0.17
    )

    ## Evaluate the model
    test_scores = model.evaluate(test['x'], test['y'], verbose=2)
    print(f'Test Loss: {test_scores[0]}')
    print(f'Test Accuracy: {test_scores[1]}')

if __name__ == '__main__':
    from pathlib import Path

    csv = Path('neuralNetworks/src/python/data/housepricedata.csv')
    keras_functional_nn(csv)