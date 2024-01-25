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
import moviepy.editor as mp

from keras.models import load_model

#추가할 것
#1. 영상 가져오는 것
video = mp.VideoFileClip("mooong.mp4")


#2. 영상과 음성 분류하여 저장 (sample = 음성, 영상은 경로 지정)
sample = video.audio.write_audiofile("test_mooong.wav")


classid = ['관심', '경계', '공포', '분리불안', '피로', '슬픔', '안정', '불안', '즐거움']


model = load_model('lstm_fcn_model_veri_loss')




#음성 결과 저장
sample = sd.s=extract_feature1(sample)
sample1 = np.array(sample)
sample2 = sample1.reshape(-1, 40, 1000, 1)
sound_result = np.argmax(model.predict(sample2), axis=-1)



#터미널에서 실행 => /content/yolov5/detect.py --weights /content/drive/MyDrive/models/weights/best.pt --img 320 --conf 0.5 --source /content/drive/MyDrive/img/dachshund.mp4

with open("rs.pickel","rb") as fi:
    image_result = pickle.load(fi)


os.remove('rs.pickel')

#[음성, 머리, 귀, 입, 몸, 꼬리]
last = sound_result.append(image_result)


model = load_model('MLP')+

last = np.array(last)
last = last.reshape(-1, 1, 6, 1)
prob = np.argmax(model.predict(last), axis=-1)

f = open('Original.txt', 'w')
f.write(str(classid[int(prob)]))
