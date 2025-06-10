#!/usr/bin/env python
# coding: utf-8
# In[1]:



from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
import torch
import os
import pickle
from scipy.stats import levy_stable
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras.layers import Dense


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint
# try:
#     devices = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(devices[0], True)
#     tf.config.experimental.set_memory_growth(devices[1], True)
# except:
#     print('未使用GPU')

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True


session = tf.compat.v1.Session(config=config)


noise_scale = np.sqrt(0.5)*0.05* 255

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
    stable_out = np.clip(stable_out, 0, 255)
    # 取整
    stable_out = np.uint(stable_out)
    return stable_out, noise  # 这里也会返回噪声，注意返回值

def stable_noise_row(row, alpha, beta=0, scale=noise_scale):  # scale修改为0.03
    #         random_state = row.name #第0张图的随机数种子就是0，第1张图的随机数种子就是1，以此类推。。。
    return stable_noise(np.asarray(row).reshape(32, 32, 3), alpha, beta, scale)[0].reshape(3072)

# train_list=[1.9,1.5,1.3]
train_list=[1]
for alpha_train in train_list:

    acc_temp = []
    auc_temp = []
    mi_temp = []
    alpha_test_temp = []
    for repeat_time in range(5):

        (X_train_temp, y_train), (X_test_temp, y_test) = cifar10.load_data()
      # In[2]:


        # In[3]:


        times = 10
        X_train = X_train_temp.reshape(50000, -1)
        X_test_f = X_test_temp.reshape(10000, -1)
        noisy = np.tile(X_train, (times,1))
        print(np.shape(noisy))
        y_tr = np.tile(y_train,(times+1,1))

        # In[4]:


        print(np.shape(noisy))
        print(np.shape(y_tr))


        # ## 小测试

        # ## 给训练集添加噪声，每个original image生成5个noisy image

        # In[6]:


        from sklearn.metrics import mutual_info_score
        if alpha_train == 0:
            average_train = np.average(X_train, axis=0)
            average_train_noisy = np.average(noisy, axis=0)
            MI = mutual_info_score(average_train_noisy, average_train)
            X_train = np.r_[X_train,noisy]
            X_train = X_train.reshape(50000*(times+1),32,32,3)
        else:
            X_noise = np.apply_along_axis(lambda x: stable_noise_row(x, alpha = alpha_train, scale=noise_scale), axis=1, arr=noisy)
            print(np.shape(X_noise))
            average_train_noisy = np.average(X_noise, axis=0)
            average_train = np.average(X_train, axis=0)
            MI = mutual_info_score(average_train_noisy, average_train)
            X_noise = X_noise.reshape(50000*times,32,32,3)
            X_train = X_train.reshape(50000,32,32,3)
            X_train = np.r_[X_train,X_noise]
            print(np.shape(X_train))

        X_train = X_train/ 255
        for j in range(3):
            X_train[:, :, :, j] = (X_train[:, :, :, j] - np.mean(X_train[:, :, :, j])) / np.std(X_train[:, :, :, j])


        # In[8]:


 


        # ## 模型建立与训练

        # In[9]:


        num_classes = 10
        encoder = LabelEncoder()
        y_te = np.tile(y_test,(times+1,1))

        y_train = encoder.fit_transform(y_tr)
        y_test = encoder.fit_transform(y_te)
        #     y_train = encoder.fit_transform(y)
        #     y_test = encoder.fit_transform(y)


        # one hot encoding
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        y_train = y_train.reshape(50000*(times+1),10)
        y_test = y_test.reshape(10000*(times+1),10)


        # In[10]:


        # 建立基于keras的cnn模型
        # Conv2D(64, 8×8) – Conv2D(128, 6×6) – Conv2D(128, 5×5) – Softmax(10)
        import keras
        from keras.layers import Dense, Conv2D, BatchNormalization, Activation
        from keras.layers import AveragePooling2D, Input, Flatten
        from tensorflow.keras.optimizers import Adam
        from keras.regularizers import l2
        from keras import backend as K
        from keras.models import Model
        from keras.datasets import cifar10
        from keras.callbacks import ModelCheckpoint, LearningRateScheduler
        from keras.callbacks import ReduceLROnPlateau


        def resnet_block(inputs, num_filters=16,
                         kernel_size=3, strides=1,
                         activation='relu'):
            x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
            x = BatchNormalization()(x)
            if (activation):
                x = Activation('relu')(x)
            return x


        # 建一个20层的ResNet网络
        def resnet_v1(input_shape):
            inputs = Input(shape=input_shape)  # Input层，用来当做占位使用

            # 第一层
            x = resnet_block(inputs)
            print('layer1,xshape:', x.shape)
            # 第2~7层
            for i in range(6):
                a = resnet_block(inputs=x)
                b = resnet_block(inputs=a, activation=None)
                x = keras.layers.add([x, b])
                x = Activation('relu')(x)
            # out：32*32*16
            # 第8~13层
            for i in range(6):
                if i == 0:
                    a = resnet_block(inputs=x, strides=2, num_filters=32)
                else:
                    a = resnet_block(inputs=x, num_filters=32)
                b = resnet_block(inputs=a, activation=None, num_filters=32)
                if i == 0:
                    x = Conv2D(32, kernel_size=3, strides=2, padding='same',
                               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
                x = keras.layers.add([x, b])
                x = Activation('relu')(x)
            # out:16*16*32
            # 第14~19层
            for i in range(6):
                if i == 0:
                    a = resnet_block(inputs=x, strides=2, num_filters=64)
                else:
                    a = resnet_block(inputs=x, num_filters=64)

                b = resnet_block(inputs=a, activation=None, num_filters=64)
                if i == 0:
                    x = Conv2D(64, kernel_size=3, strides=2, padding='same',
                               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
                x = keras.layers.add([x, b])  # 相加操作，要求x、b shape完全一致
                x = Activation('relu')(x)
            # out:8*8*64
            # 第20层
            x = AveragePooling2D(pool_size=2)(x)
            # out:4*4*64
            y = Flatten()(x)
            # out:1024
            outputs = Dense(10, activation='softmax',
                            kernel_initializer='he_normal')(y)

            # 初始化模型
            # 之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
            model = Model(inputs=inputs, outputs=outputs)
            return model


        model = resnet_v1((32, 32, 3))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        model.summary()
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1)

        # ## 给测试集加噪声，每个original image生成5个noisy

        # In[9]:


        noisy_test = np.tile(X_test_f, (times,1))


        # In[10]:


        import gc
        del X_train
        gc.collect()


        # In[11]:


        alpha_tests = [0,2,1.9,1.5,1.3,1,0.9]
        # alpha_tests=[0,2,1.9]
        for alpha_test in alpha_tests:
            print(alpha_test)
            X_test = X_test_f
            if alpha_test == 0:
                X_test = np.r_[X_test,noisy_test]
                X_test = X_test.reshape(10000*(times+1),32,32,3)
            else:
                X_noise_test = np.apply_along_axis(lambda x: stable_noise_row(x, alpha=alpha_test, scale=noise_scale),
                                              axis=1, arr=noisy_test)
                X_noise_test = X_noise_test.reshape(10000*times,32,32,3)
                X_test = X_test.reshape(10000,32,32,3)
                X_test = np.r_[X_test,X_noise_test]
            
            X_test = X_test/ 255
            for j in range(3):
                X_test[:, :, :, j] = (X_test[:, :, :, j] - np.mean(X_test[:, :, :, j])) / np.std(X_test[:, :, :, j])

                    
            y_pred = model.predict(X_test)
            tt1=np.argmax(y_test, axis=1)
            auc_value = metrics.roc_auc_score(y_test,y_pred,multi_class='ovo',average='macro')


            tt2=np.argmax(y_pred, axis=1)
            acc = metrics.accuracy_score(tt1, tt2)



            acc_temp.append(acc)
            auc_temp.append(auc_value)
            mi_temp.append(MI)
            alpha_test_temp.append(alpha_test)
            del X_test
            gc.collect()

        del model
        gc.collect()
    np.savetxt('accuracy_{}.txt'.format(alpha_train), acc_temp)
    np.savetxt('auc_{}.txt'.format(alpha_train), auc_temp)
    np.savetxt('MutualInfo_{}.txt'.format(alpha_train), mi_temp)
    np.savetxt('alpha_test__{}.txt'.format(alpha_train), alpha_test_temp)

        # print('the training alpha is %f, the testing alpha is %f, the test acc is %f, the auc is %f, the mutual information is %f'%(alpha_train, alpha_test,acc,auc_value,MI))
    #             print('the training alpha is %f, the testing alpha is %f, the test acc is %f, the auc is %f'%(alpha_train, alpha_test,acc,auc_value))


# In[ ]:




