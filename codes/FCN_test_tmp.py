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
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# %%
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


# %%
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
            alpha_c = np.random.choice(alphas)
            noise[i,j] = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 255 的置 255，低于 0 的置 0
    # stable_out = np.clip(stable_out, 0, 255)
    # 取整
    # stable_out = np.uint(stable_out)
    return stable_out, noise # 这里也会返回噪声，注意返回值
def stable_noise_mixture_hundred(inputs, alphas, beta, scale):
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
    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        alpha_c = np.random.choice(alphas, size=temp.shape)
        # 产生stable noise
        noise = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)
        # 将噪声和图片叠加
        stable_out = temp + noise
        # 将超过 255 的置 255，低于 0 的置 0
        stable_out = np.clip(stable_out, 0, 255)
        # 取整
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
alpha_test_list = [0, 2, 1.9, 1.5, 1.3, 1, 0.9, 0.5, 'mixture', 'multiple']
alpha_train_list = [2.0]
# alpha_train_list = ['multiple']
b = 64

for alpha_train in alpha_train_list:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for depth, width in [(3, 3)]:
            print(f'depth={depth}, width={width}')
            lr = 0.001
            iters = 100000
            noise_scale = np.sqrt(0.5) * sigma * 255
            model = Sequential()
            for j in range(depth-2):
                model.add(Dense(width, activation='relu',input_shape=(1,784)))
            model.add(Dense(10, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=Adam(lr=lr),
                                metrics=['accuracy'])
            ls_acc_repeat = []
            num = 0
            model.load_weights('../models/mnist_FCN_width{}_depth{}_b{}_alpha{}_sigma{}_num{}_lr{}_iter{}.h5'.format(width, depth, b, alpha_train, sigma, num, lr, iters))
            ls_acc = []
            for alpha_test in alpha_test_list:
                print('alpha=',alpha_test)
                if alpha_test == 0:
                    X_clean_test = X_te.copy()
                    X_clean_test = X_clean_test / 255.
                    X_clean_test = (X_clean_test - np.mean(np.asarray(X_clean_test))) / np.std(np.asarray(X_clean_test))
                    y_test = y_te
                    y_test = encoder.fit_transform(y_test)
                    # one hot encoding
                    # y_test = np_utils.to_categorical(y_test, num_classes)
                    X_clean_test = np.array(X_clean_test)
                    # y_test = np.array(y_test)
                    X_clean_test = X_clean_test.reshape(-1,1,784)
                    y_pred = model.predict(X_clean_test)
                    y_pred_label = np.argmax(y_pred, axis=2).flatten()
                    y_pred = y_pred.squeeze()
                    acc = accuracy_score(y_test, y_pred_label)
                    print(acc)
                    ls_acc.append(acc)
                else:
                    X_test, y_test = pickle.load(open(f"{alpha_test}_testset_0.2.txt", 'rb'))
                    X_test = X_test.reshape(-1,1,784)
                    y_pred = model.predict(X_test)
                    y_pred_label = np.argmax(y_pred, axis=2).flatten()
                    y_pred = y_pred.squeeze()
                    acc = accuracy_score(y_test, y_pred_label)
                    print(acc)
                    ls_acc.append(acc)
            ls_acc_repeat.append(ls_acc)
        with open("../results/acc_FCN_width{}_depth{}_b{}_alpha{}_sigma{}_0.2_iter100000.txt".format(width,depth,b,alpha_train,sigma), "wb") as path_acc:
            pickle.dump(ls_acc_repeat, path_acc)

# %%
from matplotlib import pyplot
depth, width = 3, 3

# plt.figure(dpi=300)
alphas = [1.0]
palette = pyplot.get_cmap('Set1')
opt_sigma = dict()
for alpha in alphas:
    i=0
    plt.figure(figsize=(10, 8),dpi=300)
    # if alpha != 'multiple' and alpha != 'mixture':
    #     alpha = float(alpha)
# for alpha in [1.9]:
    if i == 9:
        color = '#00FFFF'
    if i == 10:
        color = 'black'
    if i == 11:
        color='fuchsia'
    if alpha == 0:
        sigma_list = [0.1]
    else:
        sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_list = []
    color=palette(i)
    for sigma in sigma_list:
        color=palette(i)
        acc_path = f"../results/acc_FCN_width{width}_depth{depth}_b64_alpha{alpha}_sigma{sigma}_0.2_iter100000.txt"
        if os.path.exists(acc_path) == 0:
            print('no exists,', acc_path)
            continue
        f = open(acc_path, 'rb')
        try:
            acc = pickle.load(f)
        except:
            print("the file is empty, alpha={}".format(alpha))
            continue
        df = pd.DataFrame(acc)
        npdf = np.array(df)
        means = np.array(npdf.mean(axis=0).round(decimals=4))
        stds = np.array(npdf.std(axis=0).round(decimals=4))
        mean_list.append(np.mean(means[:-3]))
        x = ['mixture','multiple','clean','2', '1.9', '1.5', '1.3', '1', '0.9', '0.5']
        if alpha == 0:
            labels = 'clean data'
        else:
            labels = 'train alpha = {}, sigma = {}'.format(alpha, sigma)
        plt.errorbar(x[2:],means[:-2],yerr=stds[:-2],capsize=3,fmt='--.',alpha=0.8,color = color,label =labels)
        plt.axvline(x.index('0.5')-2+0.2,linestyle='--',color='black')
        plt.errorbar(x[:2],means[-2:],yerr = stds[-2:],fmt='o',capsize=4,alpha=0.8,color = color)
        i=i+1
        plt.ylim(0.3,1)
    if os.path.exists(acc_path) == 0:
        continue
    plt.legend()
    plt.show()
    tmp = pd.DataFrame({'sigma': sigma_list, 'avg_acc': mean_list})
    tmp = tmp.sort_values('avg_acc', ascending=False)
    print(tmp)
    opt_sigma[alpha] = tmp['sigma'].iloc[0]

# %%
from matplotlib import pyplot
depth, width = 3, 3

# plt.figure(dpi=300)
alphas = [2.0, 1.0]
palette = pyplot.get_cmap('Set1')
opt_sigma = dict()
for alpha in alphas:
    i=0
    plt.figure(figsize=(10, 8),dpi=300)
    # if alpha != 'multiple' and alpha != 'mixture':
    #     alpha = float(alpha)
# for alpha in [1.9]:
    if i == 9:
        color = '#00FFFF'
    if i == 10:
        color = 'black'
    if i == 11:
        color='fuchsia'
    if alpha == 0:
        sigma_list = [0.1]
    else:
        sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_list = []
    color=palette(i)
    for sigma in sigma_list:
        color=palette(i)
        acc_path = f"../results/acc_FCN_width{width}_depth{depth}_b64_alpha{alpha}_sigma{sigma}_0.2_iter100000.txt"
        if os.path.exists(acc_path) == 0:
            print('no exists,', acc_path)
            continue
        f = open(acc_path, 'rb')
        try:
            acc = pickle.load(f)
        except:
            print("the file is empty, alpha={}".format(alpha))
            continue
        df = pd.DataFrame(acc)
        npdf = np.array(df)
        means = np.array(npdf.mean(axis=0).round(decimals=4))
        stds = np.array(npdf.std(axis=0).round(decimals=4))
        mean_list.append(np.mean(means))
        x = ['mixture','multiple','clean','2', '1.9', '1.5', '1.3', '1', '0.9', '0.5']
        if alpha == 0:
            labels = 'clean data'
        else:
            labels = 'train alpha = {}, sigma = {}'.format(alpha, sigma)
        plt.errorbar(x[2:],means[:-2],yerr=stds[:-2],capsize=3,fmt='--.',alpha=0.8,color = color,label =labels)
        plt.axvline(x.index('0.5')-2+0.2,linestyle='--',color='black')
        plt.errorbar(x[:2],means[-2:],yerr = stds[-2:],fmt='o',capsize=4,alpha=0.8,color = color)
        i=i+1
        plt.ylim(0.3,1)
    if os.path.exists(acc_path) == 0:
        continue
    plt.legend()
    plt.show()
    tmp = pd.DataFrame({'sigma': sigma_list, 'avg_acc': mean_list})
    tmp = tmp.sort_values('avg_acc', ascending=False)
    print(tmp)
    opt_sigma[alpha] = tmp['sigma'].iloc[0]

# %%
from matplotlib import pyplot
depth, width = 3, 3
# acc
# plt.figure(dpi=300)
alphas = [2.0,1.0]
# alphas = [0.0,2.0,1.3,1.0,0.9]
palette = pyplot.get_cmap('Set1')
plt.figure(figsize=(10, 8),dpi=300)
i = 0
for alpha in alphas:
    # if alpha != 'multiple' and alpha != 'mixture':
    #     alpha = float(alpha)
# for alpha in [1.9]:
    color=palette(i)
    if i == 9:
        color = '#00FFFF'
    if i == 10:
        color = 'black'
    if i == 11:
        color='fuchsia'
    if alpha == 2:
        sigma = 0.3
    else:
        sigma = 0.2
    mean_list = []
    acc_path = f"../results/acc_FCN_width{width}_depth{depth}_b64_alpha{alpha}_sigma{sigma}_0.2_iter100000.txt"
    if os.path.exists(acc_path) == 0:
        print('no exists,', acc_path)
        continue
    f = open(acc_path, 'rb')
    try:
        acc = pickle.load(f)
    except:
        print("the file is empty, alpha={}".format(alpha))
        continue
    df = pd.DataFrame(acc)
    npdf = np.array(df)
    means = np.array(npdf.mean(axis=0).round(decimals=4))
    stds = np.array(npdf.std(axis=0).round(decimals=4))
    mean_list.append(np.mean(means))
    x = ['mixture','multiple','clean','2', '1.9', '1.5', '1.3', '1', '0.9', '0.5']
    if alpha == 0:
        labels = 'clean data'
    else:
        labels = 'train alpha = {}, sigma = {}'.format(alpha, sigma)
    plt.errorbar(x[2:],means[:-2],yerr=stds[:-2],capsize=3,fmt='--.',alpha=0.8,color = color,label =labels)
    plt.axvline(x.index('0.5')-2+0.2,linestyle='--',color='black')
    plt.errorbar(x[:2],means[-2:],yerr = stds[-2:],fmt='o',capsize=4,alpha=0.8,color = color)
    i=i+1
    plt.ylim(0.3,1)
    plt.legend()
plt.show()