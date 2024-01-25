# Barking, Growling, Howling, Whinning, None
# UpHead, DownHead
# UpEar, DownEar
# Open, Close, Teeth
# Sit, Stand, Lay, Front, Back
# UpTail, DownTail, MidTail

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import csv
import pandas as pd

f = open('emotion.csv', 'r')
data = csv.reader(f)
header = next(data)

features = []

for row in data:
    feat = [0, 0, 0, 0, 0, 0]
    # 음성
    if row[0] == 'Barking':
        feat[0] = 1
    elif row[0] == 'Growling':
        feat[0] = 2
    elif row[0] == 'Howling':
        feat[0] = 3
    elif row[0] == 'Whinning':
        feat[0] = 4
    
    # 머리
    if row[1] == 'UpHead':
        feat[1] = 1
    elif row[1] == 'DownHead':
        feat[1] = 2
        
    # 귀
    if row[2] == 'UpEar':
        feat[2] = 1
    elif row[2] == 'DownEar':
        feat[2] = 2
        
    # 입
    if row[3] == 'Open':
        feat[3] = 1
    elif row[3] == 'Close':
        feat[3] = 2
    elif row[3] == 'Teeth':
        feat[3] = 3
    
    # 자세
    if row[4] == 'Sit':
        feat[4] = 1
    elif row[4] == 'Stand':
        feat[4] = 2
    elif row[4] == 'Lay':
        feat[4] = 3
    elif row[4] == 'Front':
        feat[4] = 4
    elif row[4] == 'Back':
        feat[4] = 5
        
    # 꼬리
    if row[5] == 'UpTail':
        feat[5] = 1
    elif row[5] == 'DownTail':
        feat[5] = 2
    elif row[5] == 'MidTail':
        feat[5] = 3
        
    class_label = row[6]
        
    features.append([feat, class_label])
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

# class_label 값을 원-핫 인코딩 후 훈련 데이터와 검증 데이터 분리
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

import tensorflow as tf

n_columns = 6
n_row = 1
n_channels = 1
n_classes = 9

# input shape 조정
# cpu를 사용해서 수행한다
with tf.device('/cpu:0'):
    x_train = tf.reshape(x_train, [-1, n_row, n_columns, n_channels])
    x_test = tf.reshape(x_test, [-1, n_row, n_columns, n_channels])


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense, Dropout, Flatten

MLP = Sequential()
MLP.add(InputLayer(input_shape=(1, 6))) # input layer
MLP.add(Dense(256, activation='relu')) # hidden layer 1
MLP.add(Dense(256, activation='relu')) # hidden layer 2
MLP.add(Dense(256, activation='relu'))
MLP.add(Flatten())
MLP.add(Dense(9, activation='softmax')) # output layer

# summary
MLP.summary()
MLP.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
history = MLP.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

result = MLP.evaluate(x_test, y_test, batch_size=10)
result
#test loss, test acc

classid = ['관심', '경계', '공포', '분리불안', '피로', '슬픔', '안정', '불안', '즐거움']

test = [0, 1, 1, 1, 3, 0]
test = np.array(test)
test = test.reshape(-1, 1, 6, 1)
prob = np.argmax(MLP.predict(test), axis=-1)
print(str(classid[int(prob)]))