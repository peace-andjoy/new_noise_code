#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from pyts import datasets
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from scipy.stats import levy_stable
import tensorflow as tf
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# In[3]:


from pyts import datasets
(data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset("ECG200",return_X_y=True)


# In[5]:

for train_sigma in range(0.01,0.55,0.005):
    noise_scale = np.sqrt(0.5)*train_sigma 

    def stable_noise(img, alpha, beta, scale):


        noise = levy_stable.rvs(alpha=alpha, beta=beta, scale=scale, size=img.shape)

        stable_out = img + noise

        return stable_out, noise 

    def stable_noise_row(row, alpha, beta=0, scale=noise_scale): 
        return stable_noise(np.asarray(row), alpha, beta, scale)[0]

    def stable_noise_mixture(img, alphas, beta, scale):

        noise = np.empty_like(img)

        for i in range(noise.shape[0]):
            alpha_c = np.random.choice(alphas)
            noise[i] = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)

        stable_out = img + noise

        stable_out = np.clip(stable_out, 0, 255)

        stable_out = np.uint(stable_out)
        return stable_out, noise 

    def stable_noise_mixture_row(row, alphas, beta=0, scale=noise_scale): 

        return stable_noise_mixture(np.asarray(row), alphas, beta, scale)[0]


    # In[6]:


    plt.plot(data_train[1])



    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
    from tensorflow.keras.layers import MaxPooling1D, Conv1D
    from tensorflow.keras.layers import LSTM, Bidirectional
    from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, Permute, concatenate, Activation, add
    import numpy as np
    import math

    def get_model(model_name, input_shape, nb_class):
        if model_name == "vgg":
            model = cnn_vgg(input_shape, nb_class)
        else:
            print("model name missing")
        return model



    def cnn_vgg(input_shape, nb_class ,num_filter_block,nb_cnn):
        # K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
        
        ip = Input(shape=input_shape)
        
        conv = ip
        
        # nb_cnn = int(round(math.log(input_shape[0], 2))-3)

        print("pooling layers: %d"%nb_cnn)
        
        for i in range(nb_cnn):
            print('i is {}'.format(i))
            num_filters = min(num_filter_block*2**i, 512)
            conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
            conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
            if i > 1:
                conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            
        flat = Flatten()(conv)
        
        fc = Dense(4096, activation='relu')(flat)
        # fc = Dropout(0.5)(fc)
        
        fc = Dense(4096, activation='relu')(fc)
        # fc = Dropout(0.5)(fc)
        
        out = Dense(nb_class, activation='softmax')(fc)
        
        model = Model([ip], [out])
        model.summary()
        return model


    # In[18]:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_filter_block',type=int)
    parser.add_argument('--nb_cnn',type=int)
    args = parser.parse_args()
    num_filter_block=args.num_filter_block
    nb_cnn=args.nb_cnn
    tttt=int(num_filter_block)
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.python.keras.utils.np_utils import to_categorical
    from sklearn import metrics
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    model_type = 'vgg'
    train_list = ['mix','mixwithout0.5',6,5.5,0,2,1.9,1.5,1.3,1,0.9,0.5]
    alpha_trains = [2,1.9,1.5,1.3,1,0.9,0.5]
    times_tmp = 2 
    nb_iterations = 10000
    batch_size = 256


    for alpha_train in train_list:

        acc_temp = []
        auc_temp = []
        alpha_test_temp = []
        for repeat_time in range(5):

            (data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset("ECG200",return_X_y=True)

            times = 10

            X_train = data_train.copy()
            epoch_num = np.ceil(nb_iterations * (batch_size / X_train.shape[0])).astype(int)

            X_train_max = np.max(X_train)
            X_train_min = np.min(X_train)
            X_train = 2. * (X_train - X_train_min) / (X_train_max - X_train_min) - 1.

            X_test_f = data_test
            X_test_max = np.max(X_test_f)
            X_test_min = np.min(X_test_f)
            X_test_f = 2. * (X_test_f - X_test_min) / (X_test_max - X_test_min) - 1.
            
            samples_train,features = np.shape(X_train)
            samples_test = np.shape(X_test_f)[0]
            noisy = np.tile(X_train, (times,1))
            y_tr = np.tile(target_train,(1,times+1))
            y_tr = y_tr.reshape(samples_train*(times+1),1)
            num_classes = 2
            encoder = LabelEncoder()

            from sklearn.metrics import mutual_info_score
            if alpha_train == 0:

                X_train = X_train
                y_tr = target_train.copy()
                y_train = encoder.fit_transform(y_tr)
                y_train = to_categorical(y_train, num_classes)
                y_train = np.array(y_train)
            else:
                if alpha_train == 5.5:
                    num = len(alpha_trains)
                    noisy_ = np.tile(X_train, (times_tmp,1))#200*96
                    for alphas in alpha_trains:
                        temp = noisy_.copy() #100*times_tmp, 96
                        noise = np.apply_along_axis(lambda x: stable_noise_row(x, alpha = alphas, scale=noise_scale), axis=1, arr=temp)

                        X_train = np.r_[X_train,noise]
                        print(np.shape(X_train))
                    y_tra = np.tile(target_train,(1,times_tmp*num+1))
                    y_train = y_tra.reshape(-1)
                    y_train = encoder.fit_transform(y_train)
                    y_train = to_categorical(y_train, num_classes)
                    y_train = np.array(y_train)
                elif alpha_train == 6:
                    num = len(alpha_trains)
                    noisy_ = np.tile(X_train, (times_tmp,1))#200*96
                    for alphas in alpha_trains:
                        temp = noisy_.copy() #100*times_tmp, 96
                        noise = np.apply_along_axis(lambda x: stable_noise_row(x, alpha = alphas, scale=noise_scale), axis=1, arr=temp)

                        X_train = np.r_[X_train,noise]
                        print(np.shape(X_train))
                    y_tra = np.tile(target_train,(1,times_tmp*num+1))
                    y_train = y_tra.reshape(-1)
                    y_train = encoder.fit_transform(y_train)
                    y_train = to_categorical(y_train, num_classes)
                    y_train = np.array(y_train)
                    
                elif alpha_train == 'mixwithout0.5':
                    X_noise = np.apply_along_axis(lambda x: stable_noise_mixture_row(x, alphas = alpha_trains, scale=noise_scale),1,arr=noisy)
                    # X_noise = stable_noise(noisy, alpha=alpha_train, beta=0, scale=noise_scale)[0]
                    print(np.shape(X_train))
                    print(np.shape(X_noise))
                    X_train = np.r_[X_train,X_noise]

                    y_train = encoder.fit_transform(y_tr)
                    y_train = to_categorical(y_train, num_classes)
                    y_train = np.array(y_train)
                    print("the third one {}".format(np.shape(X_train)))
                    print("the forth one {}".format(np.shape(y_train)))
                elif alpha_train == 'mix':
                    X_noise = np.apply_along_axis(lambda x: stable_noise_mixture_row(x, alphas = alpha_trains, scale=noise_scale),1,arr=noisy)
                    # X_noise = stable_noise(noisy, alpha=alpha_train, beta=0, scale=noise_scale)[0]
                    print(np.shape(X_train))
                    print(np.shape(X_noise))
                    X_train = np.r_[X_train,X_noise]

                    y_train = encoder.fit_transform(y_tr)
                    y_train = to_categorical(y_train, num_classes)
                    y_train = np.array(y_train)
                    print("the third one {}".format(np.shape(X_train)))
                    print("the forth one {}".format(np.shape(y_train)))

                else:
                    X_noise = np.apply_along_axis(lambda x: stable_noise_row(x, alpha = alpha_train, scale=noise_scale),1,arr=noisy)
                    # X_noise = stable_noise(noisy, alpha=alpha_train, beta=0, scale=noise_scale)[0]
                    print(np.shape(X_train))
                    print(np.shape(X_noise))
                    X_train = np.r_[X_train,X_noise]

                    y_train = encoder.fit_transform(y_tr)
                    y_train = to_categorical(y_train, num_classes)
                    y_train = np.array(y_train)
                    print("the third one {}".format(np.shape(X_train)))
                    print("the forth one {}".format(np.shape(y_train)))

            input_shape = np.shape(X_train[0].reshape(features,1))
            # model = get_model(model_type, input_shape, num_classes)
            model = cnn_vgg(input_shape, num_classes ,num_filter_block,nb_cnn)
            es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=40, min_delta=0.01)
            model.compile(loss='categorical_crossentropy',optimizer=Adam(), metrics=['accuracy'])

            model.summary()
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_num, verbose=1,callbacks=[es])
            model.save('./model/vgg_trainalpha{}_trainsigma{}_repeattimes{}_numfilters{}_num_blocks{}.h5'.format(alpha_train,train_sigma,repeat_time,tttt,nb_cnn))