import numpy as np
import os
from pyts import datasets
import pickle
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
from tensorflow.keras.models import load_model
import gc
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# In[3]:


from pyts import datasets
(data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset("ECG200",return_X_y=True)


# In[5]:

sigma = 0.03
noise_scale = np.sqrt(0.5)*sigma 

def stable_noise(img, alpha, beta, scale):


    noise = levy_stable.rvs(alpha=alpha, beta=beta, scale=scale, size=img.shape)

    stable_out = img + noise

    stable_out = np.clip(stable_out, 0, 255)

    stable_out = np.uint(stable_out)
    return stable_out, noise 

def stable_noise_row(row, alpha, beta=0, scale=noise_scale):  
    return stable_noise(np.asarray(row).reshape(32, 32, 3), alpha, beta, scale)[0].reshape(3072)


def stable_noise_mixture(img, alphas, beta, scale):

    noise = np.empty_like(img)

    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            alpha_c = np.random.choice(alphas)
            noise[i,j] = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)

    stable_out = img + noise

    stable_out = np.clip(stable_out, 0, 255)

    stable_out = np.uint(stable_out)
    return stable_out, noise 

def stable_noise_mixture_row(row, alphas, beta=0, scale=noise_scale):  

    return stable_noise_mixture(np.asarray(row).reshape(32, 32, 3), alphas, beta, scale)[0].reshape(3072)


def stable_noise_hundred(inputs, alpha, beta=0, scale=noise_scale): 

    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        temp_noise = stable_noise(temp, alpha, beta, scale)[0]
        noisy_data[i:i+100] = temp_noise
    return noisy_data

def stable_noise_mixture_hundred(inputs, alphas, beta, scale):

    noisy_data = np.zeros(inputs.shape)
    num_samples = np.shape(inputs)[0]
    k = np.arange(0,num_samples,100)
    for i in k:
        temp = inputs[i:i+100]
        alpha_c = np.random.choice(alphas, size=temp.shape)

        noise = levy_stable.rvs(alpha=alpha_c,beta=beta,scale=scale)

        stable_out = temp + noise

        stable_out = np.clip(stable_out, 0, 255)

        stable_out = np.uint(stable_out)
        noisy_data[i:i+100] = stable_out
    return noisy_data
#

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
train_list = ['mixwithout0.5',5.5,0,2,1.9,1.5,1.3,1,0.9,0.5]
# train_list = [6]
alpha_trains = [2,1.9,1.5,1.3,1,0.9,0.5]
times_tmp = 2 #每种噪声加几倍
nb_iterations = 10000
batch_size = 256
opt_sigma = {0:0.01,2:0.2,'mixwithout0.5':0.15,5.5:0.1,1.9:0.2,1.5:0.1,1.3:0.04,1:0.05,0.9:0.01,0.5:0.02}
for alpha_train in train_list:
    train_sigma = opt_sigma[alpha_train]
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

        model = load_model('./model/vgg_trainalpha{}_trainsigma{}_repeattimes{}_numfilters{}_num_blocks{}.h5'.format(alpha_train,train_sigma,repeat_time,num_filter_block,nb_cnn)) 
## 测试集构造
        noisy_test = np.tile(X_test_f, (times,1))

## 进行测试
        alpha_tests = ['mix',6,0,2,1.9,1.5,1.3,1,0.9,0.5]
        for alpha_test in alpha_tests:
            X_test = X_test_f.copy()
            y_test = target_test.copy()
            if alpha_test == 0:
                ########ONLY REPEAT ONCE FOR CLEAN DATA TEST###########
                y_te = y_test.copy()
                y_test = encoder.fit_transform(y_te)
                y_test = to_categorical(y_test, num_classes)
                y_test = np.array(y_test)
                y_test = y_test.reshape(samples_test,num_classes)
                ########################################################
                X_test = X_test
            else:
                if alpha_test == 6:
                    noisy_t = np.tile(X_test, (times_tmp,1))#200*96
                    for alphas in alpha_trains:
                        temp = noisy_t.copy() #100*times_tmp, 96
                        noise_t = stable_noise_hundred(temp,alpha=alphas,beta=0,scale = noise_scale)
                        X_test = np.r_[X_test,noise_t]
                        print(np.shape(X_test))
                    num = len(alpha_trains)
                    y_tes = np.tile(target_test,(1,times_tmp*num+1))
                    y_test = y_tes.reshape(-1)
                    y_test = encoder.fit_transform(y_test)
                    y_test = to_categorical(y_test, num_classes)
                    y_test = np.array(y_test)
                elif alpha_test == 'mix':
                    y_te = np.tile(target_test,(1,times+1))
                    y_te = y_te.reshape(samples_test*(times+1),1)
                    y_test = encoder.fit_transform(y_te)
                    y_test = to_categorical(y_test, num_classes)
                    y_test = np.array(y_test)
                    y_test = y_test.reshape(samples_test*(times+1),num_classes)

                    X_noise_test = stable_noise_mixture_hundred(noisy_test,alphas = alpha_trains,beta=0,scale = noise_scale)
                    X_test = np.r_[X_test,X_noise_test]
                else:
                    y_te = np.tile(target_test,(1,times+1))
                    y_te = y_te.reshape(samples_test*(times+1),1)
                    y_test = encoder.fit_transform(y_te)
                    y_test = to_categorical(y_test, num_classes)
                    y_test = np.array(y_test)
                    y_test = y_test.reshape(samples_test*(times+1),num_classes)

                    X_noise_test = stable_noise_hundred(noisy_test,alpha=alpha_,beta=0,scale = noise_scale)
                    X_test = np.r_[X_test,X_noise_test]
        
            
            y_pred = model.predict(X_test)
            tt1=np.argmax(y_test, axis=1)
            auc_value = metrics.roc_auc_score(y_test,y_pred,multi_class='ovo',average='macro')


            tt2=np.argmax(y_pred, axis=1)
            acc = metrics.accuracy_score(tt1, tt2)

            acc_temp.append(acc)
            auc_temp.append(auc_value)
            alpha_test_temp.append(alpha_test)
        
        del model
        gc.collect()

    # np.savetxt('accuracy_alpha{}_modelvgg_numfilter{}.txt'.format(alphaa, num_filter_block), acc_temp)
    # np.savetxt('auc_alpha{}_modelvgg_numfilter{}.txt'.format(alphaa, num_filter_block), auc_temp)
    # np.savetxt('alpha_test_alpha{}_modelvgg_numfilter{}.txt'.format(alphaa, num_filter_block), alpha_test_temp)
    with open("./results/ECG20_VGG13_numfilters{}_b{}_alpha{}_trainsigma{}_testsigma{}_num_blocks{}.txt".format(args.num_filter_block,batch_size,alpha_train,train_sigma,sigma,nb_cnn), "wb") as path_acc:
        pickle.dump(acc_temp, path_acc)
    # with open("./results/ECG20_VGG13_numfilters{}_b{}_alpha{}_sigma{}_num_blocks{}_alphatest.txt".format(args.num_filter_block,batch_size,alpha_train,sigma,nb_cnn), "wb") as path_test:
    #     pickle.dump(alpha_test, path_test)




