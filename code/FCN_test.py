#!/usr/bin/env python
# coding: utf-8


import warnings

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from numpy.random import RandomState
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
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.set_memory_growth(devices[1], True)
except:
    print('CPU')

from tensorflow import keras
from tensorflow.keras.layers import Dense


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, help="sigma of noise", default=0.3)
parser.add_argument('--b', type=int, help="batchsize", default=64)
parser.add_argument('--width', type=int, help="width of neural network", default=3)
parser.add_argument('--depth', type=int, help="depth of neural network", default=3)
parser.add_argument('--num', type=int, help="number of experiments", default=5)
parser.add_argument('--lr', type=float, help="learning rate", default=0.001)
parser.add_argument('--alpha_train', type=str, default=2)
parser.add_argument('--alpha_test', type=str, default=2)
parser.add_argument('--sigma_train', type=float)
parser.add_argument('--sigma_test', type=float)
args = parser.parse_args()

sigma = args.sigma
b = args.b
width = args.width
depth = args.depth
experiments_num = args.num
lr = args.lr
alpha_train = args.alpha_train
alpha_test = args.alpha_test
sigma_train = args.sigma_train
sigma_test = args.sigma_test

if alpha_train == '0':
    sigma_train = 0.1

if depth != 3:
    lr = 0.0005
    iters = 50000
else:
    lr = 0.001
    iters = 10000
respath = "../results20231202/acc_FCN_width{}_depth{}_b{}_alphatrain{}_sigmatrain{}_lr{}_iter{}_alphatest{}_sigmatest{}.txt".format(width,depth,b,alpha_train,sigma_train,lr,iters,alpha_test,sigma_test)

if os.path.exists(respath):
    print('already exists!')
    import sys
    sys.exit()

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
    global rdm
    noise = levy_stable.rvs(alpha=alpha,beta=beta,scale=scale,size=img.shape,random_state=rdm)
    stable_out = img + noise
    stable_out = np.clip(stable_out, 0, 255)
    stable_out = np.uint(stable_out)
    return stable_out, noise

def stable_noise_mixture(img, alphas, beta, scale):
    '''
    input:
        img   :  image
        alphas  :  shape parameters
        beta :  symmetric parameter
        scale : scale parameter
        random_state : random seed
    return:
        stable_out : noisy image
        noise        : noise
    '''
    global rdm
    noise = np.empty_like(img)
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            alpha_c = rdm.choice(alphas)
            noise[i,j] = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale,random_state=rdm)
    stable_out = img + noise
    stable_out = np.clip(stable_out, 0, 255)
    stable_out = np.uint(stable_out)
    return stable_out, noise

def stable_noise_row(row, alpha, beta=0, scale=30):
    '''
    row : a row in pd.dataframe
    alpha, beta, scale : alpha, beta, scale of noise
    '''
    return stable_noise(np.asarray(row).reshape(28,28), alpha, beta, scale)[0].reshape(784)

def stable_noise_mixture_row(row, alphas, beta=0, scale=30):
    '''
    row : a row in pd.dataframe
    alphas, beta, scale : alphas, beta, scale of noise
    '''
    return stable_noise_mixture(np.asarray(row).reshape(28,28), alphas, beta, scale)[0].reshape(784)

def stable_noise_hundred(inputs, alpha, beta=0, scale=30):
    '''
    add single noise to 100 rows at once
    inputs: the whole dataset
    alpha, beta, scale : alphas, beta, scale of noise
    '''
    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        temp_noise = stable_noise(temp, alpha, beta, scale)[0]
        noisy_data[i:i+100] = temp_noise
    return noisy_data

def stable_noise_mixture_hundred(inputs, alphas, beta, scale):
    '''
    add mixture noise to 100 rows at once
    inputs:the whole dataset
    alpha, beta, scale : alphas, beta, scale of noise
    '''
    global rdm
    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        alpha_c = rdm.choice(alphas, size=temp.shape)
        noise = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale,random_state=rdm)
        stable_out = temp + noise
        stable_out = np.clip(stable_out, 0, 255)
        stable_out = np.uint(stable_out)
        noisy_data[i:i+100] = stable_out
    return noisy_data

# %%
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_tr, X_te = X[:60000], X[60000:]
y_tr, y_te = y[:60000], y[60000:]

# %%
aug_times_test = 10
encoder = LabelEncoder()

# %%

if depth != 3:
    lr = 0.0005
    iters = 50000
else:
    lr = 0.001
    iters = 10000
noise_scale = np.sqrt(0.5) * sigma_test * 255
model = Sequential()
for j in range(depth-2):
    model.add(Dense(width, activation='relu',input_shape=(1,784)))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=Adam(lr=lr),
                    metrics=['accuracy'])
ls_acc_repeat = []

if alpha_train != 'multiple' and alpha_train != 'mixture':
    alpha_train = float(alpha_train)
if alpha_test == '0':
    X_test = X_te.copy()
    X_test = X_test / 255.
    X_test = (X_test - 0.1307) / 0.3081
    y_test = y_te
    y_test = encoder.fit_transform(y_test)
elif alpha_test == 'mixture':
    X_test = np.tile(X_te,(aug_times_test,1))
    alphas = [2, 1.9, 1.5, 1.3, 1, 0.9]
    X_test = stable_noise_mixture_hundred(inputs=X_test, alphas=alphas, beta=0, scale=noise_scale)
    X_test = X_test / 255.
    X_test = (X_test - 0.1307) / 0.3081
    y_test = np.tile(y_te, aug_times_test)
    y_test = encoder.fit_transform(y_test)
elif alpha_test == 'multiple':
    # training set加噪声
    for alpha_test2 in [2, 1.9, 1.5, 1.3, 1, 0.9]:
        aug_times_test2 = 2 #noise扩充倍数
        X_noise_test = np.tile(X_te,(aug_times_test2,1))
        X_noise_test = stable_noise(X_noise_test, alpha = alpha_test2, beta=0, scale=noise_scale)[0]
        # X_train = clean + noise
        if alpha_test2 == 2:
            X_test = X_noise_test
        else:
            X_test = np.vstack((X_test, X_noise_test))
    X_test = X_test / 255.
    X_test = (X_test - 0.1307) / 0.3081
    y_test = np.tile(y_te, aug_times_test2*6)
    y_test = encoder.fit_transform(y_test)
else:
    alpha_test = float(alpha_test)
    X_test = np.tile(X_te,(aug_times_test,1))
    X_test = stable_noise(X_test, alpha = alpha_test, beta=0, scale=noise_scale)[0]
    X_test = X_test / 255.
    X_test = (X_test - 0.1307) / 0.3081
    y_test = np.tile(y_te, aug_times_test)
    y_test = encoder.fit_transform(y_test)

for num in range(5):
    model.load_weights('../models/mnist_FCN_width{}_depth{}_b{}_alpha{}_sigma{}_num{}_lr{}_iter{}.h5'.format(width, depth, b, alpha_train, sigma_train, num, lr, iters))
    # one hot encoding
    # y_test = np_utils.to_categorical(y_test, num_classes)
    # y_test = np.array(y_test)
    X_test = X_test.reshape(-1,1,784)
    y_pred = model.predict(X_test)
    y_pred_label = np.argmax(y_pred, axis=2).flatten()
    y_pred = y_pred.squeeze()
    acc = accuracy_score(y_test, y_pred_label)
    print(acc)
    ls_acc_repeat.append(acc)
with open(respath, "wb") as path_acc:
    pickle.dump(ls_acc_repeat, path_acc)
