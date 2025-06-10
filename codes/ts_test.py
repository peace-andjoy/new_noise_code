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

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--b', type=int, help="batchsize", default=256)
parser.add_argument('--alpha_train', type=str)
parser.add_argument('--sigma_train', type=float)
parser.add_argument('--alpha_test', type=str)
parser.add_argument('--sigma_test', type=float)
parser.add_argument('--lr', type=float, default=1e-2, help="Learning Rate")
parser.add_argument('--nb_iterations', type=int, default=10000, help="Number of iterations")
parser.add_argument('--num_units', type=int, help="numbers of units in LSTM (only needed when model_name==lstm_self)")
parser.add_argument('--num_layers', type=int, help="numbers of layers in LSTM (only needed when model_name==lstm_self)")

args = parser.parse_args()
lr = args.lr
alpha_train = args.alpha_train
sigma_train = args.sigma_train
alpha_test = args.alpha_test
sigma_test = args.sigma_test
num_units = args.num_units
num_layers = args.num_layers
nb_iterations = args.nb_iterations
batch_size = args.b

if alpha_train == 0:
    sigma_train = 0.05

try:
    alpha_train = float(alpha_train)
except:
    pass

try:
    alpha_test = float(alpha_test)
except:
    pass

savepath = "../results20231202/acc_LSTM_width{}_depth{}_b{}_alphatrain{}_sigmatrain{}_lr{}_iter{}_alphatest{}_sigmatest{}.txt".format(num_units,num_layers,batch_size,alpha_train,sigma_train,lr,nb_iterations,alpha_test,sigma_test)
if os.path.exists(savepath):
    print(savepath, 'exists!')
    import sys
    sys.exit()

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

rdm = np.random.RandomState(123)

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
    return stable_out, noise # 这里也会返回噪声，注意返回值

def stable_noise_mixture(img, alphas, beta, scale):
    '''
    此函数用将产生的stable噪声加到图片上。mixture：对每个像素随机加不同alpha的噪声。
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
    noise = np.empty_like(img)
    # 产生stable noise
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            alpha_c = rdm.choice(alphas)
            noise[i,j] = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 255 的置 255，低于 0 的置 0
    # stable_out = np.clip(stable_out, 0, 255)
    # 取整
    # stable_out = np.uint(stable_out)
    return stable_out, noise # 这里也会返回噪声，注意返回值

data = pd.read_csv("movement_libras.data", names=[i for i in range(91)])
data = data.sample(frac=1, random_state=123).reset_index(drop=True)  #shuffle
X_tr = np.asarray(data.iloc[:180, :-1])
tmp = np.asarray([X_tr[:, np.arange(0,90,2)], X_tr[:, np.arange(1,91,2)]])
X_tr = np.transpose(tmp, (1,2,0))
y_tr = np.asarray(data.iloc[:180, -1]) - 1

X_te = np.asarray(data.iloc[180:, :-1])
tmp = np.asarray([X_te[:, np.arange(0,90,2)], X_te[:, np.arange(1,91,2)]])
X_te = np.transpose(tmp, (1,2,0))
y_te = np.asarray(data.iloc[180:, -1]) - 1
nb_class = 15
input_shape = (45, 2)
# 归一化
def normalize(x):
    x_max = np.nanmax(x)
    x_min = np.nanmin(x)
    x = 2. * (x - x_min) / (x_max - x_min) - 1.
    return x

X_tr = normalize(X_tr)
X_te = normalize(X_te)

scale_test = np.sqrt(0.5) * sigma_test
aug_times_test = 10

X_te = X_te.reshape((-1, input_shape[0], input_shape[1])) 
X_tr = X_tr.reshape((-1, input_shape[0], input_shape[1])) 
y_te = np_utils.to_categorical(y_te, nb_class)
y_tr = np_utils.to_categorical(y_tr, nb_class)

if alpha_test == 0:
    X_test = X_te
    y_test = y_te
elif alpha_test == 'multiple':
    alphas =  [2, 1.9, 1.5, 1.3, 1, 0.9]
    for alpha_test2 in alphas:
        aug_times_test2 = 2 #noise扩充倍数
        X_noise_test = np.tile(X_te,(aug_times_test2,1,1))
        X_noise_test = stable_noise(X_noise_test, alpha=alpha_test2, beta=0, scale=scale_test)[0]
        # X_train = clean + noise
        if alpha_test2 == 2:
            X_test = X_noise_test
        else:
            X_test = np.vstack((X_test, X_noise_test))
    y_test = np.tile(y_te, (aug_times_test2*len(alphas),1))
elif alpha_test == 'mixture':
    X_test = np.tile(X_te,(aug_times_test,1,1))
    alphas = [2, 1.9, 1.5, 1.3, 1, 0.9]
    X_test = stable_noise_mixture(X_test, alphas=alphas, beta=0, scale=scale_test)[0]
    y_test = np.tile(y_te, (aug_times_test,1))
else:
    X_test = np.tile(X_te,(aug_times_test,1,1))
    X_test = stable_noise(X_test, alpha=alpha_test, beta=0, scale=scale_test)[0]
    y_test = np.tile(y_te, (aug_times_test,1))


import models
from tensorflow.keras.optimizers import Nadam

optm = Nadam(lr=lr)
model = models.get_model('lstm_self', input_shape, nb_class, num_units, num_layers)
model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
ls_acc_repeat = []
for num in range(5):
    try:
        modelpath = '../models/libras_LSTM_width{}_depth{}_b{}_alpha{}_sigma{}_num{}_lr{}_iter{}.h5'.format(num_units, num_layers, batch_size, alpha_train, sigma_train, num, lr, nb_iterations)
    except:
        modelpath = '../models/libras_LSTM_width{}_depth{}_b{}_alpha{}_sigma{}_num{}_lr{}_iter{}.h5'.format(num_units, num_layers, batch_size, int(alpha_train), sigma_train, num, lr, nb_iterations)
    model.load_weights(modelpath)
    acc = model.evaluate(X_test, y_test)[1]
    ls_acc_repeat.append(acc)
    # with open("../results/acc_LSTM_width{}_depth{}_b{}_alpha{}_sigma{}.txt".format(width,depth,b,alpha_train,sigma), "wb") as path_acc:
with open(savepath, "wb") as path_acc:
    pickle.dump(ls_acc_repeat, path_acc)
