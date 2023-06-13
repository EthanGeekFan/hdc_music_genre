import logging
import os
import keras.layers as layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam


def train_keras_model(input_dim, train_X, train_Y, dev_X, dev_Y, batch_size, num_epochs):

    print("Build HDC-RNN model using Keras...")
    model = Sequential()

    import pdb; pdb.set_trace()

    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(units=input_dim, activation='relu'))
    model.add(layers.Dense(units=input_dim, activation='softmax'))


    # model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
    # model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    # model.add(Dense(units=genre_features.train_Y.shape[1], activation="softmax"))

    print("Compiling ...")
    # Keras optimizer defaults:
    # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
    # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
    # SGD    : lr=0.01,  momentum=0.,                             decay=0.
    opt = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.build(train_X.shape)
    model.summary()

    print("Training ...")
    # batch_size = 20  # num of training examples per minibatch
    # num_epochs = 300
    history = model.fit(
        train_X,
        train_Y,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data = (dev_X, dev_Y)
    )

    print("\nValidating ...")
    score, accuracy = model.evaluate(
        dev_X, dev_Y, batch_size=batch_size, verbose=1
    )
    print("Dev loss:  ", score)
    print("Dev accuracy:  ", accuracy)


    # print("\nTesting ...")
    # score, accuracy = model.evaluate(
    #     genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1
    # )
    # print("Test loss:  ", score)