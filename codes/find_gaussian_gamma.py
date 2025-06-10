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
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--b', type=int, help="batchsize", default=64)
parser.add_argument('--width', type=int, help="width of neural network", default=3)
parser.add_argument('--depth', type=int, help="depth of neural network", default=3)
parser.add_argument('--num', type=int, help="number of experiments", default=5)
parser.add_argument('--lr', type=float, help="learning rate", default=0.001)
parser.add_argument('--iter', type=int, help="number of iterations", default=10000)
args = parser.parse_args()

b = args.b
width = args.width
depth = args.depth
experiments_num = args.num
lr = args.lr

def stable_noise(img, alpha, beta, scale):
    '''
    此函数用将产生的stable噪声加到图片上
    传入:
        img   :  原图
        alpha  :  shape parameter
        beta :  symmetric parameter
        scale : scale parameter
        random_state : 随机数种子
    返回:
        stable_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 产生stable noise
    noise = levy_stable.rvs(alpha=alpha,beta=beta,scale=scale,size=img.shape)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 255 的置 255，低于 0 的置 0
    stable_out = np.clip(stable_out, 0, 255)
    # 取整
    stable_out = np.uint(stable_out)
    return stable_out, noise # 这里也会返回噪声，注意返回值

def stable_noise_hundred(inputs, alpha, beta=0, scale=30):
    '''
    对数据集中的行添加噪声
    每一百行一起添加噪声
    inputs:整个数据集
    alpha, beta, scale : 需要添加噪声的alpha, beta, scale
    '''
    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        temp_noise = stable_noise(temp, alpha, beta, scale)[0]
        noisy_data[i:i+100] = temp_noise
    return noisy_data

def gaussian_noise(img, sigma):
    '''
    此函数用将产生的stable噪声加到图片上
    传入:
        img   :  原图
        alpha  :  shape parameter
        beta :  symmetric parameter
        scale : scale parameter
        random_state : 随机数种子
    返回:
        stable_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 产生stable noise
    noise = np.random.normal(0, sigma*255, size=img.shape)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 255 的置 255，低于 0 的置 0
    stable_out = np.clip(stable_out, 0, 255)
    # 取整
    stable_out = np.uint(stable_out)
    return stable_out, noise # 这里也会返回噪声，注意返回值

# %%
for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print('sigma=', sigma)
    for num in range(experiments_num):
        print('experiment {}'.format(num))
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X_tr, X_te = X[:60000], X[60000:]
        y_tr, y_te = y[:60000], y[60000:]
        # 计算noise的scale
        X_train = X_tr
        aug_times_train = 10
        X_noise = np.tile(X_tr,(aug_times_train,1))
        X_noise = gaussian_noise(X_noise, sigma)[0]
        # X_train = clean + noise
        X_train = np.vstack((X_tr, X_noise))
        X_train = np.asarray(X_train) / 255.
        X_train = (X_train - np.mean(X_train)) / np.std(X_train)

        # y_train:
        y_train = np.tile(y_tr, aug_times_train+1)

        print(np.shape(X_train))
        print(np.shape(y_train))

        num_classes = 10
        encoder = LabelEncoder()

        y_train = encoder.fit_transform(y_train)

        # one hot encoding
        y_train = np_utils.to_categorical(y_train, num_classes)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(np.shape(X_train))
        print(np.shape(y_train))

        X_train = X_train.reshape(-1,1,784)
        y_train = y_train.reshape(-1,1,10)

        print(np.shape(X_train))
        print(np.shape(y_train))

        num_classes = 10

        model = Sequential()
        # model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation="relu", input_shape=(28, 28, 1) ))
        for i in range(depth-2):
            model.add(Dense(width, activation='relu',input_shape=(1,784)))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=Adam(lr=lr),
                        metrics=['accuracy'])

        nb_iterations = args.iter
        batch_size = b
        nb_epochs = np.ceil(nb_iterations * (batch_size / X_train.shape[0])).astype(int)
        print('num_iterations:', nb_iterations, 'num_epochs:', nb_epochs)
        print(model.summary())
        model.fit(X_train, y_train, batch_size = b, epochs = nb_epochs)
        if not os.path.exists('../models'):
            os.makedirs('../models')
        model.save_weights('../models/mnist_FCN_width{}_depth{}_b{}_gaussian_sigma{}_num{}_lr{}_iter{}.h5'.format(width, depth, b, sigma, num, lr, nb_iterations))

print("end!")
# %%
