#!/usr/bin/env python
# coding: utf-8

import warnings

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from tensorflow.python.keras.utils import np_utils
from keras.models import Sequential
from sklearn.metrics import accuracy_score, roc_auc_score, mutual_info_score
from sklearn.preprocessing import LabelEncoder
import torch
import os
import pickle
from scipy.stats import levy_stable

try:
    devices = tf.config.experimental.list_physical_devices('GPU')
    print(devices)
except:
    print('CPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
parser.add_argument('--iter', type=int, help="number of iterations", default=10000)
parser.add_argument('--alpha', type=str, help="alpha of training noise")
parser.add_argument('--beta', type=float, default=0)
args = parser.parse_args()

b = args.b
width = args.width
depth = args.depth
experiments_num = args.num
lr = args.lr
alpha_train = args.alpha
beta_train = args.beta
sigma = args.sigma

rdm = np.random.RandomState(123)

def stable_noise(img, alpha, beta, scale):
    '''
    input:
        img   :  image
        alpha  :  shape parameter
        beta :  symmetric parameter
        scale : scale parameter
        random_state : random seed
    return:
        stable_out : noisy image
        noise        : noise
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
noise_scale = np.sqrt(0.5) * sigma * 255
print('sigma=', sigma)
nb_iterations = args.iter
try:
    alpha_train = float(alpha_train)
except:
    pass

for num in range(experiments_num):
    print('experiment {}'.format(num))
    savepath = '../models/mnist_FCN_width{}_depth{}_b{}_alpha{}_beta{}_sigma{}_num{}_lr{}_iter{}.h5'.format(width, depth, b, alpha_train, beta_train, sigma, num, lr, nb_iterations)
    if os.path.exists(savepath):
        print(savepath, 'already exists!')
        continue
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_tr, X_te = X[:60000], X[60000:]
    y_tr, y_te = y[:60000], y[60000:]

    if alpha_train == 'mixture':
        aug_times_train = 10
        X_noise = np.tile(X_tr,(aug_times_train,1))
        alphas = [2, 1.9, 1.5, 1.3, 1, 0.9]
        X_noise = stable_noise_mixture_hundred(inputs=X_noise, alphas=alphas, beta=beta_train, scale=noise_scale)
        X_train = np.vstack((X_tr, X_noise))
    elif alpha_train == 'multiple':
        tmp = [2, 1.9, 1.5, 1.3, 1, 0.9]
        X_train = X_tr
        for alpha_ in tmp:
            aug_times_train = 2 #noise扩充倍数
            X_noise = np.tile(X_tr,(aug_times_train,1))
            X_noise = stable_noise_hundred(X_noise, alpha = alpha_, beta=beta_train, scale=noise_scale)
            # X_train = clean + noise
            X_train = np.vstack((X_train, X_noise))
        aug_times_train = aug_times_train * len(tmp)
    else:
        alpha_train = float(alpha_train)
        # 计算noise的scale
        X_train = X_tr
        aug_times_train=0
        if alpha_train != 0:
            aug_times_train = 10
            X_noise = np.tile(X_tr,(aug_times_train,1))
            X_noise = stable_noise_hundred(X_noise, alpha=alpha_train, beta=beta_train, scale=noise_scale)
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
    nb_epochs = np.ceil(nb_iterations * (b / X_train.shape[0])).astype(int)
    print('num_iterations:', nb_iterations, 'num_epochs:', nb_epochs)
    print(model.summary())
    model.fit(X_train, y_train, batch_size = b, epochs = nb_epochs)
    if not os.path.exists('../models'):
        os.makedirs('../models')
    model.save_weights(savepath)

print("end!")