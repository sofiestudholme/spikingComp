"""
Helper funcs for doing MNIST with PNN
2 x 2 kernel method
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from snn_helper import *


try:
    from keras.models import Sequential 
    import tensorflow as tf
    from keras.layers import Activation, Dense
    import keras
except:
    print('cant import keras')

def load_mnist_data(return_split=False):
    """
    usage:
    X,X_train,X_test,Y_test,Y_train = load_mnist_data(True)
    X,y = load_mnist_data()
    """
    X, label_mnist = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )

    X_train = X[0:60000,:]
    X_test = X[60000:70000]
    Y_test = label_mnist[60000:70000]
    Y_train  = label_mnist[0:60000]
    if return_split:
        return X,X_train,X_test,Y_test,Y_train
    else:
        return X,label_mnist


def transform_image(image,network,num_filters=16,only_outputs=False,single_transform=0,shape=(28,28),width=2):
    """ image is vector, will reshape
    out image is 2D matrix 
    pass in num_filters, the number of outputs from the convolution
    if single transform != 0, will choose a single instance of event rates for patterns that is always the same, rather than a random transform
    """
    image = image.reshape(shape)
    out_image = np.empty((num_filters,shape[0]-(width-1),shape[0]-(width-1)))
    for row in range(shape[0]-(width-1)):
        for col in range(shape[0]-(width-1)):
            input = image[row:row+width,col:col+width]
            out = network[str(input.reshape(-1))]
            is_vec=True if len(out.shape)==1 else False
            if only_outputs:
                out = out[np.random.randint(out.shape[0]),[4,5,6,7,12,13,14,15]]
            else:
                if not is_vec:
                    out = out[np.random.randint(out.shape[0]),:]
            out_image[:,row,col] = out
    return out_image

def get_transformed_x(X,channels,network,output_size=27*27,input_shape=(28,28),save='',**kwargs):
    """
    transform X using network 
    need to have network dict defined
    input X is MNIST digits 
    channels: dimension of filter, one for each readout current / event train / event rate. 
    output_size: number of pixels in each output channel. For 2*2 kernel with stride 1, this is 27,27.
    input_shape = shape of input image. Normally (28,28) unless downsampled.
    needs function transform_image.
    if save is a string, will save image with that title
    """
    num_test_instances = 70000
    X_new = np.empty((num_test_instances,output_size*channels)).astype('float32')
    for idx, image in enumerate(X[0:num_test_instances,:]):
        out_image = transform_image(image>255/2,network,channels, shape=input_shape,**kwargs)
        X_new[idx,:] = out_image.reshape(-1)
    if save != '':
        np.save(save,X_new)
    return X_new

def plot_accuracy(a):
    plt.plot(a.history['accuracy'])
    plt.plot(a.history['val_accuracy'])
    plt.title('D3 and D4')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()

#############################################
########## workflow #########################
############################################3

########################### imports #####################################

# from mnist_helper import *
# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload


identifier = "MNIST_CTL_7_R100_V10_nonP5a_SR00_ER500000_IR5000_X400_Y400_input_groups_[ 153 2678 5861 4794 1515]_output_groups_[6578]_T_input_100000_output_voltage_groups[8, 777, 3607, 5386, 5962, 6176, 4415, 863, 6, 2640]_seq.csv"

def train_model(idenfitier): 

    ########################### #read in file ###############################
    path = f"results/RC_board/Readout_voltage/400"
    currents, y , voltages = read_in_file(path,identifier=identifier,info=False,start_index=0,end_index=100000*8*18,num_input_voltages=4)


    ############################# params ##################################
    input_T=100000
    tau=0
    downsample_rate=input_T
    threshold=1
    max_plot_len=100000

    ############################ downsample to get x,y  ##################
    e = get_LIF_readout_from_current(currents,tau,threshold)
    y,l=get_patterns_as_int(voltages,show_patterns=False,return_lookup=True)
    x1 =downsample_flexible(e,downsample_rate) # shape of x is downsampled_timesteps, channnels
    y1=downsample_flexible(y,downsample_rate,class_vector=True)
    max_plot_len = np.rint(max_plot_len / downsample_rate)

    plot_classes_PCA(x1,y1,legend=1,lookup=l,triangles=False)

    # get network dict -> can combine more than one dataset
    network={}
    for p in range(16):
        network[str(L4[p].astype(bool))]=np.concatenate((x1[y1==p],x2[y2==((p+1)%16)]),axis=1)
    big_x = network[str(L4[p].astype(bool))]

    # transform X, save if want
    X,_ = load_mnist_data()
    x = get_transformed_x(X,14,network,save='mnist_voltages_d6_d7')

    # load X_transformed
    X_new = (np.load("mnist_voltages_d3_d4.npy")).astype('float32')


    # train model

    _,_,_,Y_test,Y_train = load_mnist_data(True)
    epochs = 40
    model = Sequential() 
    layer_1 = Dense(10, input_shape = (21*2*27*27,),activation='softmax') 
    model.add(layer_1) 
    model.summary()

    model.compile('rmsprop',keras.losses.categorical_crossentropy,metrics=["accuracy"])
    true = tf.one_hot(Y_train,10)
    a = model.fit(
        X_new[0:60000,:],true,epochs=epochs,validation_split=0.2
    )

    score = model.evaluate(X_new[60000:70000,:], tf.one_hot(Y_test,10), verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])