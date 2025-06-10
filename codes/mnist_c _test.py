# %%
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 数据下载，分割训练集测试集
import warnings

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.metrics import accuracy_score, roc_auc_score, mutual_info_score
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
import torch
import os
import pickle
from scipy.stats import levy_stable

try:
    devices = tf.config.experimental.list_physical_devices('GPU')
    print(devices)
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.set_memory_growth(devices[1], True)
except:
    print('未使用GPU')

# 神经网络模型搭建
from tensorflow import keras
from tensorflow.keras.layers import Dense


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import json
import math
import os
import sys
import time

import tensorflow.compat.v1 as tf

import numpy as np

# %%
# alpha_sigma_train_list = [(0.0, 0.1), (2.0, 0.25), (1.9, 0.35), (1.5, 0.2), (1.3, 0.15), (1.0, 0.1), (0.9, 0.1), (0.5, 0.05), ('mixture', 0.25), ('multiple', 0.25)]
alpha_sigma_train_list = [('multiple', 0.25)]
# alpha_train_list = ['multiple']
b = 64

for alpha_train, sigma_train in alpha_sigma_train_list:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1), padding="same"))
    model.add(MaxPooling2D((2,2), padding="same"))
    model.add(Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D((2,2), padding="same"))
    model.add(Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=Adam(lr=1e-4),
                    metrics=['accuracy'])
    model.load_weights('../models/mnist_c_b64_alpha{}_sigma{}_num0_lr0.0001_iter100000.h5'.format(alpha_train, sigma_train))
    encoder = LabelEncoder()
    record = []
    for pack in os.listdir('mnist_c'):
        X_te = np.load('mnist_c/'+pack+'/test_images.npy')
        y_te = np.load('mnist_c/'+pack+'/test_labels.npy')
        X_test = X_te.copy()
        X_test = (X_test - np.mean(np.asarray(X_test))) / np.std(np.asarray(X_test))
        y_test = y_te
        y_test = encoder.fit_transform(y_test)
        # one hot encoding
        # y_test = np_utils.to_categorical(y_test, num_classes)
        X_test = np.array(X_test)
        # y_test = np.array(y_test)
        y_pred = model.predict(X_test)
        y_pred_label = np.argmax(y_pred, axis=1).flatten()
        y_pred = y_pred.squeeze()
        acc = accuracy_score(y_test, y_pred_label)
        print(pack, acc)
        record.append((pack, acc))
    dic = dict(record)
    print(alpha_train, sigma_train)
    for key in ['shot_noise', 'impulse_noise', 'glass_blur', 'motion_blur', 'shear', 'scale', 'rotate', 'brightness', 'translate', 'stripe', 'fog', 'spatter', 'dotted_line', 'zigzag', 'canny_edges', 'identity']:
        print(dic[key], end=' ')

    # clean
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_te = X[60000:]
    y_te = y[60000:]
    X_test = X_te.copy()
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_test = (X_test - np.mean(np.asarray(X_test))) / np.std(np.asarray(X_test))
    y_test = y_te
    y_test = encoder.fit_transform(y_test)
    # one hot encoding
    # y_test = np_utils.to_categorical(y_test, num_classes)
    X_test = np.array(X_test)
    # y_test = np.array(y_test)
    y_pred = model.predict(X_test)
    y_pred_label = np.argmax(y_pred, axis=1).flatten()
    y_pred = y_pred.squeeze()
    acc = accuracy_score(y_test, y_pred_label)
    print('clean', acc)
# %%
