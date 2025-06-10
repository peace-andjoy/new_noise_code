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

import multiprocessing as mp
from tqdm import tqdm

# %%
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_tr, X_te = X[:60000], X[60000:]
y_tr, y_te = y[:60000], y[60000:]

# %%
def stable_noise2(img, alpha, beta, scale):
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
    return stable_out

# %%
aug_times_test = 10
encoder = LabelEncoder()
sigma = 0.2
noise_scale = np.sqrt(0.5) * sigma * 255

# %%
for alpha_test in [2,1.9,1.5,1.3,1,0.9,0.5]:
    for beta in [0.2, -0.2, 0.5, -0.5, 0.8, -0.8, 1, -1]:
        if os.path.exists(f"testset/assymetric/alpha{alpha_test}_beta{beta}_testset_{sigma}.txt"):
            continue
        X_noise_test = np.tile(X_te,(aug_times_test,1))
        pool = mp.Pool(max(int(mp.cpu_count()/2), 16))  # 创建进程池，使用CPU核心数/2作为进程数
        res = [pool.apply_async(stable_noise2, args=(img, alpha_test, beta, noise_scale)) for img in X_noise_test]
        noisy = []
        for p in tqdm(res):
            noisy.append(p.get())
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有子进程完成
        X_noise_test = np.asarray(noisy)
        X_test = np.vstack((X_te, X_noise_test))
        X_test = X_test / 255.
        X_test = (X_test - np.mean(X_test)) / np.std(X_test)
        y_test = np.tile(y_te, aug_times_test+1)
        y_test = encoder.fit_transform(y_test)
        fw = open(f"testset/assymetric/alpha{alpha_test}_beta{beta}_testset_{sigma}.txt", "wb")
        pickle.dump((X_test, y_test), fw)