# %%
from scipy.io import arff
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import levy_stable
from keras.utils import np_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
import pickle
import models
import gc
import os
import sys
# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--b', type=int, help="batchsize", default=256)
parser.add_argument('--width', type=int, help="width of neural network(only needed when model_name==FCN)", default=3)
parser.add_argument('--depth', type=int, help="depth of neural network(only needed when model_name==FCNb)", default=3)
parser.add_argument('--num', type=int, help="number of experiments", default=5)
parser.add_argument('--alpha', type=str)
parser.add_argument('--sigma', type=float)
parser.add_argument('--model_name', type=str, default='mlp4')
parser.add_argument('--lr', type=float, default=1e-2, help="Learning Rate")
parser.add_argument('--nb_iterations', type=int, default=1500, help="Number of iterations")
parser.add_argument('--optimizer', type=str, default="sgd", help="Which optimizer")
parser.add_argument('--dataset', type=str, default="pendigits", help="Which dataset")
parser.add_argument('--num_units', type=int, help="numbers of units in LSTM (only needed when model_name==lstm_self)")
parser.add_argument('--num_layers', type=int, help="numbers of layers in LSTM (only needed when model_name==lstm_self)")

args = parser.parse_args()
width = args.width
depth = args.depth
experiments_num = args.num
alpha = args.alpha
sigma = args.sigma
model_name = args.model_name
num_units = args.num_units
num_layers = args.num_layers
nb_iterations = args.nb_iterations
batch_size = args.b

# %%
def get_data(filepath):
    #读取数据，返回x（只有一个维度）和y
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    # 样本
    sample = df.values[:, 0:len(df.values[0])-1]
    # 对标签进行处理
    # [b'1' b'-1' ...]bytes类型
    label = df.values[:, -1] # 要处理的标签
    cla = [] # 处理后的标签
    for i in label:
        test = int(i)
        cla.append(test)
    y = np.asarray(cla)
    x = np.asarray(df.iloc[:,:-1])
    return x, y

def combine_x(x_1, x_2):
    #合并两个维度的x
    x_1 = np.expand_dims(x_1, axis=2)
    x_2 = np.expand_dims(x_2, axis=2)
    x = np.concatenate([x_1, x_2],axis=2)
    return x

def FCN(input_shape, nb_class, width, depth):
    # Z. Wang, W. Yan, T. Oates, "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline," Int. Joint Conf. Neural Networks, 2017, pp. 1578-1585
    
    ip = Input(shape=input_shape)
    fc = Flatten()(ip)
    for i in range(depth-1):
        fc = Dense(width, activation='relu')(fc)
    out = Dense(nb_class, activation='softmax')(fc)
    model = Model([ip], [out])
    model.summary()
    return model

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
    noise = levy_stable.rvs(alpha=alpha, beta=beta, scale=scale, size=img.shape)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    # stable_out = np.clip(stable_out, 0, 255)
    # 取整
    # stable_out = np.uint(stable_out)
    return stable_out, noise  # 这里也会返回噪声，注意返回值

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
# %%
try:
    alpha = float(alpha)
except:
    pass

for num in range(experiments_num):
    savepath = '../models/libras_LSTM_width{}_depth{}_b{}_alpha{}_sigma{}_num{}_lr{}_iter{}.h5'.format(num_units, num_layers, batch_size, alpha, sigma, num, args.lr, nb_iterations)
    if os.path.exists(savepath):
        print(savepath, 'already exists!')
        continue
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

    scale = np.sqrt(0.5) * sigma

    if alpha == "multiple":
        X_te = X_te.reshape((-1, input_shape[0], input_shape[1])) 
        X_tr = X_tr.reshape((-1, input_shape[0], input_shape[1])) 
        y_te = np_utils.to_categorical(y_te, nb_class)
        y_tr = np_utils.to_categorical(y_tr, nb_class)
        X_train = X_tr
        lena = 0
        for alpha_ in [2, 1.9, 1.5, 1.3, 1, 0.9]:
            aug_times_train = 2 #noise扩充倍数
            X_noise = np.tile(X_tr,(aug_times_train,1,1))
            X_noise = stable_noise(X_noise, alpha=alpha_, beta=0, scale=scale)[0]
            # X_train = clean + noise
            X_train = np.vstack((X_train, X_noise))
            lena += 1
        y_train = np.tile(y_tr, (aug_times_train*lena+1, 1))
    elif alpha == "mixture":
        # training set加噪声
        aug_times_train = 10 #noise扩充倍数
        X_noise = np.tile(X_tr,(aug_times_train,1,1))
        alphas = [2, 1.9, 1.5, 1.3, 1, 0.9]
        X_noise = stable_noise_mixture(X_noise, alphas=alphas, beta=0, scale=scale)[0]
        X_train = np.vstack((X_tr, X_noise))
        y_train = np.tile(y_tr, aug_times_train+1)
        X_te = X_te.reshape((-1, input_shape[0], input_shape[1]))
        X_tr = X_tr.reshape((-1, input_shape[0], input_shape[1])) 
        y_te = np_utils.to_categorical(y_te, nb_class)
        y_train = np_utils.to_categorical(y_train, nb_class)
    else:
        if alpha == 0:
            aug_times_train = 0
            X_train = X_tr
            y_train = y_tr
        else:
            # training set加噪声
            aug_times_train = 10 #noise扩充倍数
            X_noise = np.tile(X_tr,(aug_times_train,1,1))
            X_noise = stable_noise(X_noise, alpha=alpha, beta=0, scale=scale)[0]
        # X_train = clean + noise
            X_train = np.vstack((X_tr, X_noise))
            y_train = np.tile(y_tr, aug_times_train+1)
        X_te = X_te.reshape((-1, input_shape[0], input_shape[1])) 
        X_train = X_train.reshape((-1, input_shape[0], input_shape[1])) 
        y_te = np_utils.to_categorical(y_te, nb_class)
        y_train = np_utils.to_categorical(y_train, nb_class)

    if model_name == 'FCN':
        model = FCN(input_shape, nb_class, width, depth)
    else:
        model = models.get_model(model_name, input_shape, nb_class, num_units, num_layers)
    if args.optimizer=="adam":
        from tensorflow.keras.optimizers import Adam
        optm = Adam(lr=args.lr)
    elif args.optimizer=="nadam":
        from tensorflow.keras.optimizers import Nadam
        optm = Nadam(lr=args.lr)
    elif args.optimizer=="adadelta":
        from tensorflow.keras.optimizers import Adadelta
        optm = Adadelta(lr=args.lr, rho=0.95, epsilon=1e-8)
    else:
        from tensorflow.keras.optimizers import SGD
        optm = SGD(lr=args.lr, decay=5e-4, momentum=0.9) #, nesterov=True)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    nb_epochs = np.ceil(nb_iterations * (batch_size / X_train.shape[0])).astype(int)
    print(y_train.shape)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs)
    model.save_weights(savepath)

print('end!')