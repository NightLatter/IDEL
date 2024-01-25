import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization, Input, Convolution2D, Activation,TimeDistributed
from keras.layers import Input, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import GlobalAveragePooling1D, Permute, Dropout
from keras.utils import to_categorical
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import librosa
import pickle
from tensorflow.keras import layers 
from sklearn.preprocessing import LabelEncoder
import sound

sd = sound.sound_marge

featuresdf = sd.sound_learning()

# 피클로 데이터 저장
featuresdf.to_pickle("featuresdf.pkl")

# 피클 데이터 로드
featuresdf = pd.read_pickle("featuresdf.pkl")

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

n_columns = 174
n_row = 40       
n_channels = 1
n_classes = 10

# input shape 조정
# cpu를 사용해서 수행한다
with tf.device('/cpu:0'):
    x_train = tf.reshape(x_train, [-1, n_row, n_columns, n_channels])
    x_test = tf.reshape(x_test, [-1, n_row, n_columns, n_channels])
    

model = sd.generate_model()
model.summary()

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
model.fit(x_train, y_train, epochs=13000, batch_size=16)
model.save("lstm_fcn_model_2")