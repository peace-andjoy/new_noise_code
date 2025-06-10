from scipy.io import arff
import pandas as pd
import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt
from scipy.stats import levy_stable
from keras.utils import np_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
import pickle
import models
import gc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, help="sigma of noise", default=0.03)
parser.add_argument('--b', type=int, help="batchsize", default=256)
parser.add_argument('--width', type=int, help="width of neural network", default=3)
parser.add_argument('--depth', type=int, help="depth of neural network", default=3)
parser.add_argument('--num', type=int, help="number of experiments", default=5)
parser.add_argument('--alpha_list', nargs='+',default=[0, 2, 1.9, 1.5, 1.3, 1, 0.9, 0.5, 'multiple', 'mixture'])
parser.add_argument('--model_name', type=str, default='mlp4')
parser.add_argument('--lr', type=float, default=1e-2, help="Learning Rate")
parser.add_argument('--optimizer', type=str, default="sgd", help="Which optimizer")
parser.add_argument('--dataset', type=str, default="pendigits", help="Which dataset")
parser.add_argument('--num_units', type=int, help="numbers of units in LSTM (only needed when model_name==lstm_self)")
parser.add_argument('--num_layers', type=int, help="numbers of layers in LSTM (only needed when model_name==lstm_self)")

args = parser.parse_args()

width = args.width
depth = args.depth
experiments_num = args.num
alpha_list = args.alpha_list
model_name = args.model_name
num_units = args.num_units
num_layers = args.num_layers

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
    global rdm
    noise = levy_stable.rvs(alpha=alpha, beta=beta, scale=scale, size=img.shape, random_state=rdm)
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
    global rdm
    noise = np.empty_like(img)
    # 产生stable noise
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            alpha_c = rdm.choice(alphas)
            noise[i,j] = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale, random_state=rdm)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 255 的置 255，低于 0 的置 0
    # stable_out = np.clip(stable_out, 0, 255)
    # 取整
    # stable_out = np.uint(stable_out)
    return stable_out, noise # 这里也会返回噪声，注意返回值

for alpha in alpha_list:
    print('alpha=', alpha)
    try:
        alpha = float(alpha)
    except:
        pass
    ls_acc_repeat = []
    for num in range(experiments_num):
        print('model {}'.format(num))
        if args.dataset == 'pendigits':
            filepath_d1_train = 'PenDigits/PenDigitsDimension1_TRAIN.arff'
            filepath_d1_test = 'PenDigits/PenDigitsDimension1_TEST.arff'
            filepath_d2_train = 'PenDigits/PenDigitsDimension2_TRAIN.arff'
            filepath_d2_test = 'PenDigits/PenDigitsDimension2_TEST.arff'
            X_tr_1 = get_data(filepath_d1_train)[0]
            X_tr_2 = get_data(filepath_d2_train)[0]
            X_tr = combine_x(X_tr_1, X_tr_2)

            X_te_1 = get_data(filepath_d1_test)[0]
            X_te_2 = get_data(filepath_d2_test)[0]
            X_te = combine_x(X_te_1, X_te_2)

            y_tr = get_data(filepath_d1_train)[1]
            y_te = get_data(filepath_d1_test)[1]
            nb_class = 10
            input_shape = (8, 2)

        elif args.dataset == 'libras':
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

        sigma = args.sigma
        scale = np.sqrt(2) * sigma
        aug_times_test = 10

        X_te = X_te.reshape((-1, input_shape[0], input_shape[1])) 
        X_tr = X_tr.reshape((-1, input_shape[0], input_shape[1])) 
        y_te = np_utils.to_categorical(y_te, nb_class)
        y_tr = np_utils.to_categorical(y_tr, nb_class)

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

        batch_size = args.b
        model.load_weights('../models/libras_LSTM_width{}_depth{}_b{}_alpha{}_sigma{}_num{}.h5'.format(num_units, num_layers, batch_size, alpha, sigma, num))
        ls_acc = []
        x_clean_test = X_te
        # one hot encoding
        # y_test = np_utils.to_categorical(y_test, num_classes)
        print('clean')
        acc = model.evaluate(x_clean_test, y_te)[1]
        ls_acc.append(acc)

        y_test = np.tile(y_te, (aug_times_test+1,1))
        rdm = RandomState(num)
        for alpha_test in [2,1.9,1.5,1.3,1,0.9,0.5]:
            X_noise_test = np.tile(X_te,(aug_times_test,1,1))
            X_noise_test = stable_noise(X_noise_test, alpha=alpha_test, beta=0, scale=scale)[0]
            X_test = np.vstack((X_te, X_noise_test))
            # # one hot encoding
            print('alpha=',alpha_test)
            acc = model.evaluate(X_test, y_test)[1]
            print(acc)
            ls_acc.append(acc)

        # test on mixture noisy testset
        X_noise_test = np.tile(X_te,(aug_times_test,1,1))
        alphas = [2, 1.9, 1.5, 1.3, 1, 0.9]
        X_noise_test = stable_noise_mixture(X_noise_test, alphas=alphas, beta=0, scale=scale)[0]
        X_test = np.vstack((X_te, X_noise_test))
        # # one hot encoding
        print('mixture')
        acc = model.evaluate(X_test, y_test)[1]
        print(acc)
        ls_acc.append(acc)

        # test on multiple noisy testset
        X_test = X_te
        # training set加噪声
        for alpha_test2 in [2, 1.9, 1.5, 1.3, 1, 0.9]:
            aug_times_test2 = 2 #noise扩充倍数
            X_noise_test = np.tile(X_te,(aug_times_test2,1,1))
            X_noise_test = stable_noise(X_noise_test, alpha=alpha_test2, beta=0, scale=scale)[0]
            # X_train = clean + noise
            X_test = np.vstack((X_test, X_noise_test))
        print('multiple')
        y_test = np.tile(y_te, (aug_times_test2*len(alphas)+1,1))
        acc = model.evaluate(X_test, y_test)[1]
        print(acc)
        ls_acc.append(acc)
        
        ls_acc_repeat.append(ls_acc)
        del model
        gc.collect()

    if model_name == 'FCN':
        with open("../results/{}_acc_ts_FCN_width{}_depth{}_b{}_alpha{}_sigma{}.txt".format(args.dataset,width,depth,batch_size,alpha,sigma), "wb") as path_acc:
            pickle.dump(ls_acc_repeat, path_acc)
    elif model_name == 'lstm_self':
        with open("../results/{}_lstm_width{}_depth{}_b{}_alpha{}_sigma{}.txt".format(args.dataset,num_units,num_layers,batch_size,alpha,sigma), "wb") as path_acc:
            pickle.dump(ls_acc_repeat, path_acc)

print('end!')