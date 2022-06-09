import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split

INT = 5
OUT = 1
per = 1
per_v = 0.15

f = np.loadtxt("inputfile_2.txt", dtype=np.float32, delimiter="\t")

train_x_full = f[0:int(per*len(f)), :-1]
train_y_full = f[0:int(per*len(f)), -1]

train_x, test_x, train_y, test_y = train_test_split(train_x_full, train_y_full, test_size=0.2, shuffle=False)
# print(train_x.shape)
# print(test_x)
# print(train_y)
train_x = np.expand_dims(train_x, 0).transpose(1,2,0)
train_y = np.expand_dims(train_y, 0).transpose(1,0)
scaler = StandardScaler()

#---define model---
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import tensorflow as tf


class TCNN(tf.keras.Model):
    def __init__(self):
        super(TCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=2,
            strides=1,
            padding='same',
            dilation_rate=1,
            groups=1,
            activation='relu')
            # input_shape=(None, 153,5 ))
        self.fl = tf.keras.layers.Flatten()
        self.ds1 = tf.keras.layers.Dense(256)
        self.ds2 = tf.keras.layers.Dense(500)
        self.outputs = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.fl(x)
        x = self.ds1(x)
        x = self.ds2(x)
        return self.outputs(x)

model = TCNN()
# print(model.shape)
model.compile(optimizer='adam', loss='mae')

    #
    # ##---fit model---
model.fit(train_x, train_y, epochs=1000, batch_size=4, steps_per_epoch=5)
# input_shape = (4, 10, 128)
# x = tf.random.normal(input_shape)
#
# print(x)
