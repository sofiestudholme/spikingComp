"""

helper functions for SNN / computing with switching events
+ Logical operations

"""


##################################################
########## imports ###############################
##################################################


import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except:
    print("no pandas :(")

import csv

from IPython.display import set_matplotlib_formats

try:
    from sklearn import decomposition
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    # from statsmodels.tsa.api import SimpleExpSmoothing,ExponentialSmoothing
    from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                                mean_squared_error, plot_confusion_matrix)
    from sklearn.preprocessing import OneHotEncoder
except:
    print('Missing some imports')
from matplotlib import colors as mcolors
from collections import OrderedDict

#########################################################################
################ Defaults ###############################################
#########################################################################

# plot_defaults
PLOT=0
TRANSPARENCY = 1

# Read in data
START_INDEX=200000
END_INDEX=0

# Event trains, readout etc
EXTRA_TIME_SCALE=0
TYPE = 'e'
COMPETE_FOR_EVENTS = False
COMPETE_MODE = 'absolute'
THRESHOLD=0.0001
INPUT_T=10000
TAU=0

L = {0: np.array([0, 0, 0], dtype=int),
 1: np.array([0, 0, 1], dtype=int),
 2: np.array([0, 1, 0], dtype=int),
 3: np.array([0, 1, 1], dtype=int),
 4: np.array([1, 0, 0], dtype=int),
 5: np.array([1, 0, 1], dtype=int),
 6: np.array([1, 1, 0], dtype=int),
 7: np.array([1, 1, 1], dtype=int)}

L4 = {0: np.array([0, 0, 0, 0], dtype=int),
 1: np.array([0, 0, 0, 1], dtype=int),
 2: np.array([0, 0, 1, 0], dtype=int),
 3: np.array([0, 0, 1, 1], dtype=int),
 4: np.array([0, 1, 0, 0], dtype=int),
 5: np.array([0, 1, 0, 1], dtype=int),
 6: np.array([0, 1, 1, 0], dtype=int),
 7: np.array([0, 1, 1, 1], dtype=int),
 8: np.array([1, 0, 0, 0], dtype=int),
 9: np.array([1, 0, 0, 1], dtype=int),
 10: np.array([1, 0, 1, 0], dtype=int),
 11: np.array([1, 0, 1, 1], dtype=int),
 12: np.array([1, 1, 0, 0], dtype=int),
 13: np.array([1, 1, 0, 1], dtype=int),
 14: np.array([1, 1, 1, 0], dtype=int),
 15: np.array([1, 1, 1, 1], dtype=int)}

#########################################################################
################ LIF neuron class #######################################
#########################################################################

class neuron:
    def __init__(self,tau,threshold,leak=1,v_rest=0,membrane_resistance=1,refractory_period = 100):
      self.tau=tau
      self.threshold = threshold
      self.membrane_resistance = membrane_resistance
      self.membrane_potential = 0
      self.a = math.exp(-1/self.tau)
      self.spiked=0
      self.refractory_period = refractory_period
      self.refractory_count = 0
    
    def update_membrane_potential(self, input):
        self.spiked=0
        if self.refractory_count < 1:
            self.membrane_potential = input*self.membrane_resistance + (self.membrane_potential - input*self.membrane_resistance)*self.a
            if self.membrane_potential > self.threshold:
                self.membrane_potential = 0
                self.spiked = 1
                self.refractory_count=self.refractory_period
        self.refractory_count = self.refractory_count-1

################################################################################################
############################################## Reading in data #################################
################################################################################################


# from snn_helper import *
def read_in_file(relative_path,identifier='seq',info=False,start_index=START_INDEX,end_index=0,num_input_voltages=4,output_type='floating_voltages',start_x=0,end_x=0): 
    """
    currents, y, voltages = read_in_file(path,identifier,info=False,start_index=START_INDEX,end_index=0,switch_rate_return=False,debug=False,num_input_voltages=4)

    ISSUE WITH Y AND L:
    get_patterns_as_int from starts at 200000. From now on only takes voltages 

    Output type can be:
    floating voltages - look for V_out_{group}
    currents - look for input and output current I_{group}
    OR specify start_x and end_x and manually choose output. 

    currents output shape is (channels,samples)
    voltages output shape is (channels,samples)

    """
    results_dir=Path(os.getcwd()).parent.parent.parent / relative_path
    filename = [el for el in os.listdir(results_dir) if identifier in el and 'lock' not in el][-1]
    results_path = results_dir / filename
    data = pd.read_csv(str(results_path))

    if info:
        print(f'Filename:{filename}')
        print("Data: ")
        print(data)
    # # get rid of init data
    if start_index:
        data=remove_entries(data,0,start_index)

    # get rid of end index
    if end_index:
        data=remove_entries(data,end_index,len(data))

    if info:
        print(relative_path)
        print(f'new length of data {len(data)}')

    # same train / test data
    # data=data.append(data)
    # currents shape should be (channels,timesteps)
    if start_x==0:
        if info: print(f"Getting groups from identifier, type_ = {output_type}") 
        currents = np.array([np.array(data.loc[:,input_group]) for input_group in get_groups_from_identifier(identifier,type_=output_type)])
    else:
        currents=np.array(data.iloc[:,start_x:end_x]).T

    voltages = np.array([np.array(data.loc[:,input_group]) for input_group in get_groups_from_identifier(identifier,type_='voltages')][0:num_input_voltages])
    y = get_patterns_as_int(voltages,show_patterns=info)

    if info:
        print(data.head())
        print("\n\nCurrents / voltages_out :")
        print(currents)
        print("\n\nVoltages:")
        print(voltages)
        try:
            switch =np.array(data.loc[:,'switch'])
        except: 
            switch = np.array(data.loc[:,'switch'])
        switch = np.where(np.isnan(switch),0,switch)
        sw_rate = np.sum(switch)/len(switch)
        print(f'switching rate = {sw_rate}')

    return currents, y, voltages



def get_groups_from_identifier(identifier,type_):
    """ return groups from identifier string as list of ints  
    type_ can be either:
    - voltages - input voltages, find from input groups
    - currents - input and output currents, find from input groups
    - floating_voltages - find from output voltage groups
    """

    if type_ == "floating_voltages":
        pre="output_voltage_groups"
        assert "output_voltage_groups" in identifier,"Can't find output voltage groups"
        s = 'V_out_'
    elif type_ == "voltages" :
        pre = "input_groups"
        s = 'V_'
    elif type_ == "currents":
        pre = "input_groups"
        s = 'I_'
    else:
        raise Exception("Can't find specified type")

    pre_i = identifier.find(pre)
    g= identifier[identifier.find('[',pre_i)+1:identifier.find(']',pre_i)]
    g=g.split(' ')
    groups = [int(el.replace(',','')) for el in g if el != '']
    groups=[ s + str(g) for g in groups]
    return groups


def get_vectors_containing(d,string):
    vals = d.columns.values.tolist()
    idxs = [idx for idx in range(len(vals)) if string in vals[idx]]
    outs = []
    for idx in idxs:
        outs.append(np.array(d.loc[:,vals[idx]]))
    return outs

def remove_nan(x):
    x=np.where(np.isnan(x),0,x)
    return x




###############################################################
############ all SW events from TG ############################
###############################################################
class sw_density_info:
    def __init__(self,path):
        self.path=path
        ever_switched_d = get_all_sw_events(path)
        tgs = list(ever_switched_d.keys())
        tgs.sort()
        self.tgs = tgs # list of all tunnel gaps that ever switched

    def get_sw_density_matrix(self,start=0,end=999999999,normalize=False):
        """ return a square matrix of sw density, to make square, add zeros to fill in end of matrix 
        """
        sw_density= self.get_sw_density_vector(start,end,normalize)
        target_len = math.ceil(math.sqrt(len(sw_density)))**2
        square_sw_density = np.zeros((target_len,1))
        square_sw_density[0:len(sw_density)] = np.array(sw_density).reshape((len(sw_density),1))
        width = int(math.sqrt(target_len))
        sw_density = square_sw_density.reshape((width,width))
        return sw_density

    def get_sw_density_vector(self,start=0,end=999999999,normalize=False,vector_to_normalise_by=[]):
        d = get_all_sw_events(self.path,start,end)
        sw_density = []
        for tg in self.tgs:
            try:
                sw_density.append(d[tg])
            except:
                sw_density.append(0)
    
        if normalize:
            if vector_to_normalise_by == []:
                sw_density= np.array(sw_density) / np.array(self.get_sw_density_vector())
            else:
                sw_density= np.array(sw_density) / np.array(vector_to_normalise_by)
        return (sw_density)

    def show_sw_density(self,start=0,end=9999999999,save=False,title='sw',fig_ax=[],normalize=False):
        sw_density=self.get_sw_density_matrix(start,end,normalize=normalize)
        if fig_ax == []:
            f = plt.figure(figsize=(6.2, 5.6))
            ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
            axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])  
            if not normalize:
                im = ax.matshow(sw_density, norm=mcolors.LogNorm(vmin=1, vmax=np.max(sw_density)))
            else:
                im = ax.matshow(sw_density, norm=mcolors.LogNorm(vmin=0,vmax=np.max(sw_density)))

            f.colorbar(im, cax=axcolor)
        else:
            [f,ax] = fig_ax
            if not normalize:
                im = ax.matshow(sw_density, norm=mcolors.LogNorm(vmin=1, vmax=np.max(sw_density)))
            else:
                # print(sw_density)
                im = ax.matshow(sw_density,vmax=1/20,vmin=0) #norm=mcolors.LogNorm(vmin=0.00000000000000000000000000001,vmax=np.max(sw_density)
        if save:
            plt.savefig('images/sw_events/' + title + ".png")
        if fig_ax==[]:
            plt.show()

def get_all_sw_events(relative_path,start_index=0,end_index=99999999999999999999999):
    """ returns a dict of sw events from UpEventsRLE and DnEventsRLE csv files in results dir
    dict key, value pairs are  tunnel gap number, number of times switched
    optionally specify start and end index, values for dict in timesteps
     """
    results_dir=Path(os.getcwd()).parent.parent / relative_path
    sw_events = {}
    up_events_filename = [el for el in os.listdir(results_dir) if 'UpEventsRLE' in el]
    dn_events_filename = [el for el in os.listdir(results_dir) if 'DnEventsRLE' in el]
    
    if (len(up_events_filename) + len(dn_events_filename)) != 2:
        raise Exception("Can't find file")
    
    # up events
    with open(results_dir/up_events_filename[0], newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in file:
            x=[int(el) for el in row[0].split(',')]
            timestep=x[0]
            if timestep > start_index and timestep < end_index:
                for tg in x[1:]:
                    if tg in sw_events:
                        sw_events[tg] += 1
                    else:
                        sw_events[tg] = 1
    # same thing for dn events
    with open(results_dir/dn_events_filename[0], newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in file:
            x=[int(el) for el in row[0].split(',')]
            timestep=x[0]
            if timestep > start_index and timestep < end_index:
                for tg in x[1:]:
                    if tg in sw_events:
                        sw_events[tg] += 1
                    else:
                        sw_events[tg] = 1   
    return sw_events


def plot_state_difference(voltages,current,pattern_for_comparison,start_index,end_index,tau=1000,pass_in_y=False,pass_in_mp=False,input_T=10000):
    """ 
    from Maas, Natschlager 2002 - state difference in neural trajectories to test separation property
    calcualte the difference between state x (membrane potential) for different input patterns 
    pass in voltages array (timesteps, input_voltages)
    and current array 
    from current, obtain LPF event train readout (membrane potential)
    pattern for comparision corresponds to pattern number encoded by y to compare to other combonations
    only works with T_input=10000
    if pass in y, pass in binary truth instead of voltages
    if pass in mp, pass in membrane potential instead of current
    """

    f = plt.figure(figsize=(12,7))

    # get x and y
    if pass_in_mp:
        mp=current # mp should be shape (timesteps, channels)
    else:
        mp = get_LIF_readout_from_current(current,tau,0.0001)
    
    if not pass_in_y:
        y,lookup = get_patterns_as_int(voltages,return_lookup=True,voltages_in=True)
        num_patterns = max(y)

        
    else: 
        y = voltages
        lookup = {
            0 : 'False',
            1 : 'True'
        }
        pattern_for_comparison=0
        num_patterns=2

    y_cropped = y[start_index:end_index] 
    mp_cropped = mp[start_index:end_index,:] 
    y_1 = mp_cropped[y_cropped==pattern_for_comparison,:]
    for pattern in range(num_patterns):
        if pattern != pattern_for_comparison:
            y_2 = mp_cropped[y_cropped==pattern,:]
            if y_1.shape != y_2.shape:
                y_1 = y_1[:y_2.shape[0],:]
                y_2 = y_2[:y_1.shape[0],:]

            difference = np.subtract(y_1,y_2)
            # av_distance = np.mean(difference,axis=1)
            av_distance = np.linalg.norm(difference,axis=1)
            plt.plot(abs(av_distance),label=lookup[pattern])
    y_1_1 = mp_cropped[y_cropped==1,:]
    # difference between self at next input seq
    y_cropped = y[start_index+16*input_T:end_index+16*input_T]
    mp_cropped = mp[start_index+16*input_T:end_index+16*input_T,:] 
    y_2 = mp_cropped[y_cropped==pattern_for_comparison,:]

    # if y_1.shape != y_2.shape:
    #         y_1 = y_1[:y_2.shape[0],:]
    #         y_2 = y_2[:y_1.shape[0],:]
            
    difference = np.subtract(y_1,y_2)
    av_distance = np.linalg.norm(difference,axis=1)
    l_ = "Self " + str(lookup[pattern_for_comparison])
    plt.plot(abs(av_distance),label=l_,color='black',linewidth=2)

    if pass_in_y:
        # difference between true and true at next input seq
        y_2 = mp_cropped[y_cropped==1,:]
        y_1 = y_1_1
        if y_1.shape != y_2.shape:
                y_1 = y_1[:y_2.shape[0],:]
                y_2 = y_2[:y_1.shape[0],:]
                
        difference = np.subtract(y_1,y_2)
        av_distance = np.linalg.norm(difference,axis=1)
        l_ = "Self " + str(lookup[1])
        plt.plot((av_distance),label=l_,linewidth=2)


    # plt.ylim(0,0.02)
    plt.ylabel("$||X_{pattern} - X_{other}||$")
    plt.xlabel("timesteps")

    plt.legend(loc=4)
    plt.show()



##########################################################################
################# Currents and event trains ##############################
##########################################################################
def get_dI(current):
    if len(current.shape) == 1:
        delta_I = np.diff(current)
        delta_I = np.append(delta_I,delta_I[0])
        
    else:
        delta_I = (np.diff(current,axis=1))
        delta_I = np.append(delta_I,(delta_I[:,0]).reshape((delta_I.shape[0],1)),axis=1)
    return delta_I
    
def lif_membrane_potential(events,tau):
    """
    if tau = 0, return events vector.
    """
    Vmembrane=np.empty(events.shape)
    Vmembrane[0]=0
    if tau == 0:
        Vmembrane=events
    else:
        a=math.exp(-1/tau)
        for i in range(0,len(events)-1):
            Vmembrane[i+1]=events[i]*(1-a) + Vmembrane[i]*a
    return Vmembrane


def simple_event_train(currents,threshold,abs_delta_I = []):
    """ takes current vector, no compete for events """
    if len(abs_delta_I) == 0:
        abs_delta_I = abs(np.diff(currents))
    events = np.greater(abs_delta_I,threshold) 
    events = np.concatenate((events,np.array([0])))
    return events


def simple_event_train_2(abs_delta_I,threshold):
    """ takes abs delta I or V vector, no compete for events """
    events = np.greater(abs_delta_I,threshold) 
    return events


def event_train_array(currents,threshold,compete=1,compete_mode='absolute',threshold_array=[],debug_event_train_array=False,threshold_upper=-1):

    ''' 
    returns same shape as currents, (channels, samples) 
    if compete for events, modes can be:
    relative abs(delta I / Inew) 
    relative_1 abs(delta I / Iprevious)
    relative_2 abs(delta I / abs_delta_I_average)
    absolute
    if not compete for events, can have a threshold array for each channel or a single threshold 
    if threshold_lower is defined, introduce a threshold band instead

    '''

    abs_delta_I = abs(np.diff(currents,axis=1))
    abs_delta_I=np.c_[abs_delta_I,np.zeros((abs_delta_I.shape[0]))]

    if debug_event_train_array:
        if compete:
            print(f" Compete mode = {compete_mode}")
        else:
            print(f"Not competing for events. Threshold = {threshold}, threshold_array = {threshold_array},threshold_lower={threshold_upper}")

    # get events BEFORE competition
    if len(threshold_array)==0:   # all thresholds are the same
            if threshold_upper==-1:
                events = np.greater(abs_delta_I,threshold)
            else:
                events = np.logical_and(np.greater(abs_delta_I,threshold), np.less(abs_delta_I,threshold_upper))
    else: # array of thresholds for all the event trains
            events = np.zeros(abs_delta_I.shape)
            for i,channel in enumerate(abs_delta_I):
                events[i,:]=channel>threshold_array[i]
    events = np.multiply(events, 1)   
          
    # print(f"adi shape = {abs_delta_I.shape}")
    if compete:
        switching_occured=np.where(np.sum(events,axis=0)==0,0,1)
        if compete_mode == 'absolute':
            events = np.where(switching_occured, np.argmax(abs_delta_I,axis=0),0)
        elif compete_mode== 'relative':
            events = np.where(switching_occured, np.argmax(abs(np.divide(abs_delta_I,currents)),axis=0),0)
        elif compete_mode== 'relative_1':
            rel_2_d = currents[:,1:currents.shape[1]] + currents[:,0:currents.shape[1]-1]
            rel_2_d = np.c_[np.zeros((rel_2_d.shape[0])),rel_2_d] 
            rel_2_d = rel_2_d / 2
            events = np.where(switching_occured, np.argmax(abs(np.divide(abs_delta_I,rel_2_d)),axis=0),0)
        elif compete_mode == 'relative_2':
            b=abs_delta_I
            events = np.where(switching_occured, np.argmax(abs( np.divide(b.T,np.mean(b,axis=1)).T),axis=0),0)
        elif compete_mode == 'relative_3':
            b=currents
        
            events = np.where(switching_occured, np.argmax(abs( np.divide(abs_delta_I.T,np.mean(b,axis=1)).T    ),axis=0),0)
        else:
            raise Exception("Compete mode doesnt exist") 

        # print(f'current shape = {currents.shape}, events shape = {events.shape}')
        n_values=(currents.shape[0])
        event_train=(np.eye(n_values)[events])
        events=np.multiply(event_train,switching_occured.reshape(len(switching_occured),1))
        events = events.T 
        

    return events



def get_LIF_readout_from_current(currents,tau,threshold=THRESHOLD,compete_mode=COMPETE_MODE,compete_for_events=COMPETE_FOR_EVENTS,extra_time_scales=EXTRA_TIME_SCALE,type_=TYPE,debug=False,**kwargs):
    """ 
    returns matrix of LPF readouts from current matrix x. Output shape is (timesteps, number_channels)
    If tau = 0, returns matrix of event trains
    """
    start=1
    factors=[0.01,0.1] # factors to multiply time constant by
    len_=currents.shape
    len_=len_[1]
    x = np.empty([len(currents),len_])

    if debug:
        print(f"input={type_},threshold={threshold},compete_for_events={compete_for_events},compete_mode={compete_mode},extra_time_scales={extra_time_scales}")
    if type_ == 'e':
        trains=event_train_array(currents,threshold,compete_for_events,compete_mode=compete_mode,**kwargs)
        for train in trains:
            if start==1:
                start=0
                x = lif_membrane_potential(train,tau).reshape(1,len_)
            else:
                x = np.concatenate((x,lif_membrane_potential(train,tau).reshape(1,len_)))
            if extra_time_scales:
                for factor in factors:
                    out = lif_membrane_potential(train,tau*factor)
                    out = out.reshape((1,len(out)))
                    x = np.concatenate((x,out))
    else:
        for current in currents:
            if type_ == 'd_i':
                out = get_dI(current)
                out = lif_membrane_potential(out,tau).reshape((1,-1))

            elif type_ == 'abs_d_i':
                out = abs(get_dI(current))
                out = lif_membrane_potential(out,tau).reshape((1,-1))

            elif type_ == 'i':
                out = current[0:len(current)].reshape((1,-1))

            elif type_ == 'abs_i':
                out = abs(current[0:len(current)]).reshape((1,-1))
            else:
                raise Exception('type not valid')

            if start==1:
                start=0
                x = out
            else:
                x = np.concatenate((x,out))
    x = x.T # size is now (timesteps,readouts)
    return x

################################################################################################
###################################### RC Helper funcs #########################################
################################################################################################

def split_into_train_test(x,y):
    samples = len(y)
    X_train = x[0:(samples//2),:]
    X_test = x[(samples//2):samples,:]
    y_train = y[0:(samples//2)]
    y_test = y[(samples//2):samples]
    return X_train,X_test,y_train,y_test


def get_wta_output(probs,input_T):
    winner=np.empty(len(probs))
    for start_index in range(0,len(probs),input_T):
        end_index = len(probs)
        if end_index > start_index + input_T:
            end_index = start_index+input_T
        averages=np.sum(probs[start_index:end_index],axis=0)
        result = np.where(averages == np.amax(averages))
        winner[start_index:end_index] = result
    return winner


def remove_duplicates(a,input_T):
    a=a[::input_T]
    return a

def get_patterns_as_int(voltages,show_patterns=0,return_lookup=0):
    """
    get binary patterns from Vin as vector of ints. Optionally returns lookup dict of 
    patterns vs encoding int
    voltages is an array of voltages shape (timesteps, columns)
    """
    num_inputs=len(voltages)
    data =(np.array(voltages)).astype(int)
    data = data / np.max(data)
    inputs = np.unique(data,axis=1).T.astype(np.int8)
    input_vector = np.array(data).astype(np.int8).T
   
    y_out = np.zeros(len(input_vector),dtype=np.int8)
    if show_patterns:
        print("Patterns are:")        
    lookup = dict([])

    for i,pattern in enumerate(inputs):
        if show_patterns:
            print(f'{np.multiply(pattern,1)} = {i}')
        if return_lookup:
            lookup[i] = (np.multiply(pattern,1))
        y = np.multiply(np.sum(np.equal(input_vector,pattern),axis=1)==num_inputs,i)
        y_out = np.add(y_out,y)
    if return_lookup == 0:
        return y_out
    else:
        return y_out,lookup


##################################################################################
##################### Which cases are failing ####################################
##################################################################################
def get_sucess_rates(true,pred,patterns,l):
    """ Sucess rates by input pattern """
    correct_idx = true == pred
    patterns_correct = np.where(correct_idx,patterns,-6)

    sucess_rates = np.zeros(len(l))
    instances = np.zeros(len(l))
    correct = np.zeros(len(l))
    i=0
    for pattern_num in range(0,len(l)):
        
        denom = np.sum(patterns==pattern_num)
        c = np.sum(patterns_correct == pattern_num)
        s = -5000 if denom == 0 else c/denom 
        sucess_rates[i]=s
        instances[i]=denom
        correct[i] = c
        i+=1
    return sucess_rates, instances.astype(int), correct.astype(int)

def get_sucess_rates_df(l,true,out,patterns):
    """ Return dataframe with sucess rates by pattern """
    sucess_rates, instances, correct = get_sucess_rates(true,out,patterns,l)
    df = pd.DataFrame()
    df = df.append(l, ignore_index=True)
    df = df.append(pd.DataFrame(sucess_rates).T, ignore_index=True)
    df = df.append(pd.DataFrame(instances).T, ignore_index=True)
    df = df.set_axis(['pattern','sucess rate', 'instances'], axis=0, inplace=False).T
    df= df.drop(df[df["instances"]==0].index)
    return df



#####################################################################################
####################### Downsample with and without averaging #######################
#####################################################################################
def downsample(x,downsample_rate):
    """ 
    Downsample by sampling function every downsample_rate timesteps 
    other values are discarded 
    """
    x=np.roll(x,-10) # wait  a bit to start sampling
    x=x[::downsample_rate]
    return x


def downsample_flexible(x,downsample_rate,mode='average',class_vector = 0):
    """ 
    DOWNSAMPLE BY getting average (default) max, min EVERY DOWNSAMPLE_RATE TIMESTEPS
    x should be either vector or matrix of shape (readings, channels) 
    set class_vector to true if vector is vector of ints representing classes
    mode can be:
    average 
    max - maximum reading in downsample_rate
    min - minimum reading in downsample_rate
    single - single reading 
    return shape is (readings downsampled, channels)
    """
    timesteps = len(x)
    is_vector = len(x.shape) == 1
    if is_vector:
        x = x.reshape(1,timesteps)
    else:
        x=x.T   # shape is now (channels, readings)
    
    x_downsampled=np.empty((int(timesteps/downsample_rate),x.shape[0]))

    for row, j in zip(x,range(x.shape[0])):
        for start_index,i in zip(range(0,timesteps,downsample_rate),range(int(timesteps/downsample_rate))):
            end_index = timesteps
            if end_index > start_index + downsample_rate:
                end_index = start_index+downsample_rate
            if mode == 'average' or mode == '':
                average=np.mean(x[j,start_index:end_index])
            elif mode == 'max':
                average=np.max(x[j,start_index:end_index])
            elif mode == 'min':
                average=np.min(x[j,start_index:end_index])
            elif mode == 'single':
                average=x[j,end_index-1]
            else:
                raise Exception("Mode is wrong")    
            if class_vector:
                average = round(average)
            x_downsampled[i,j] = average
    if is_vector:
        x_downsampled=x_downsampled.reshape((len(x_downsampled)))
    return x_downsampled




############################# remove parts of data ###############################
def remove_entries(data,start_i,end_i,T_delete = 0,replace = 0):
    """ 
    takes DF object
    remove parts of the data passed in. Can be first couple of timeconstants for the sequence, reset sections etc
    Params: 
    - start_i and end_i: start and end index for deleting eg (0, 2tau) for first 2 time constats, (T_delete/2,T_delete for every second input)
    - T_delete: period of deleting, if deleting every second input this is 2*T_input, if only removing start values this is len(data) defaults to 0 -> not periodic
    - replace: replace with 0 - set this to true if want to replace removed entries with 0. Default is to remove them
    """
    data.index=range(len(data))

    if np.remainder(len(data),T_delete):
        raise Exception("delete period must be divisible by length of data")
   
    if T_delete:
        for period_start in range(0,len(data),T_delete):
            data=data.drop(range(period_start+start_i,period_start+end_i),axis=0)

    else:
        data=data.drop(range(start_i,end_i),axis=0)

    return data



def remove_entries_faster(data,start_i,end_i,T_delete = 0):
    """ 
    takes array of shape (timesteps, channels)
    remove parts of the data passed in. Can be first couple of timeconstants for the sequence, reset sections etc
    Params: 
    - start_i and end_i: start and end index for KEEPING eg (2tau,input_T) for first 2 time constats, (T_delete/2,T_delete for every second input)
    - T_delete: period of deleting, if deleting every second input this is 2*T_input, if only removing start values this is len(data) defaults to 0 -> not periodic
    """
    is_vec=False
    if len(data.shape)==1:
        is_vec=True
    if not is_vec:
        [timesteps,channels] = data.shape

        if np.remainder(timesteps,T_delete):
            raise Exception("delete period must be divisible by length of data")
        
        num_delete_periods = timesteps/T_delete 
        new_T = end_i-start_i
        new_number_of_timesteps = int(num_delete_periods * new_T)

        new_array = np.empty((new_number_of_timesteps,channels))
    
        if T_delete:
            for new_start,period_start in zip(range(0,new_number_of_timesteps,new_T),range(0,timesteps,T_delete)):
                new_array[new_start:new_start+new_T,:]=data[period_start+start_i:period_start+end_i,:]
        else:
            new_array=data[start_i:end_i,:]
    else: 
        timesteps = len(data)

        if np.remainder(timesteps,T_delete):
            raise Exception("delete period must be divisible by length of data")
        
        num_delete_periods = timesteps/T_delete 
        new_T = end_i-start_i
        new_number_of_timesteps = int(num_delete_periods * new_T)

        new_array = np.empty((new_number_of_timesteps))
    
        if T_delete:
            for new_start,period_start in zip(range(0,new_number_of_timesteps,new_T),range(0,timesteps,T_delete)):
                new_array[new_start:new_start+new_T]=data[period_start+start_i:period_start+end_i]
        else:
            new_array=data[start_i:end_i]

    return new_array



def shuffle_currents_and_voltages(currents,voltages,input_T):
    """
    currents.shape = (channels,timesteps)
    voltages.shape = (v_in,timesteps)

    return same shape arrays but shuffled by input_T
    """
    r = currents.reshape(8,input_T,-1)
    print(r.shape)
    p = np.random.permutation(int(len(v1)/input_T))
    voltages = voltages.reshape(4,input_T,-1)
    print(voltages.shape)

    voltages = voltages[:,:,p]
    currents = r[:,:,p]

    currents=currents.reshape(8,-1)
    voltages=voltages.reshape(4,-1)
    return currents,voltages



############################################ LDA, logistic regression and linear regression ###################################

def do_LDA(currents,y,tau=TAU,threshold=THRESHOLD,input_T=INPUT_T,plot=PLOT,confusion_matrix_=PLOT,max_plot_len=2000000,downsample_rate=0,input_type=[''],input_mp=False,split=True,debug=False,**kwargs):
    ''' 
    type can be e - event train, d_i - delta I, abs_d_i - abs delta I
    i -> current, no LPF
    abs_i -> abs current, no LPF
    if downsample_rate != 0, inputs can be average, max, min over the downsample time. 
    plot can be: 0 (don't plot), 1 plot all, 2 plot for binary, 3: just print accuracy
    if input_mp is true, currents = membrane potential 
    if split, split into half train, half test. If not, test and train on same data.

    accuracy, classifier, x,y = do_LDA(currents,y,tau=TAU,threshold=THRESHOLD,input_T=INPUT_T,plot=PLOT,confusion_matrix_=PLOT,max_plot_len=2000000,downsample_rate=0,input_type=[''],input_mp=False,split=True,debug=False)
    '''
    # input T scaled by downsample rate
    # errors and warnings
    if downsample_rate:
        if np.remainder(input_T,downsample_rate):
            raise Exception("input period must be divisible by downsample rate")
        else:
            input_T = int(input_T/downsample_rate)
    else:
        if input_type != ['']:
            print("Downsample rate is 0 so input type will be ignored")

    if not input_mp:
        membrane_potential = get_LIF_readout_from_current(currents,tau,threshold,debug=debug,**kwargs)
    else:
        membrane_potential=currents

    if downsample_rate:
        x =downsample_flexible(membrane_potential,downsample_rate,input_type[0]) # shape of x is downsampled_timesteps, channnels
        for mode in input_type[1:]:
            reading=downsample_flexible(membrane_potential,downsample_rate,mode) # shape of x is downsampled_timesteps, channnels
            x=np.concatenate((x,reading),axis=1)
        y=downsample_flexible(y,downsample_rate,class_vector=True)
        max_plot_len = np.rint(max_plot_len / downsample_rate)
    else:
        x=membrane_potential
    
    if split:
        X_train,X_test,y_train,y_test=split_into_train_test(x,y)
    else:
        X_train = x
        X_test=x
        y_train=y
        y_test =y
    if debug:
        print(f'Tau = {tau}, threshold = {threshold}, input_T(scaled) = {input_T}, input type = {input_type}')

  # LDA classifier
    classifier = LinearDiscriminantAnalysis(n_components=1)
    classifier.fit(X_train, y_train)
    prob_test = classifier.predict_proba(X_test)
    if input_T == 1:
        winner_test=classifier.predict(X_test)
    else:
        winner_test = get_wta_output(prob_test,input_T)

    wta_accuracy_test= np.sum(np.equal(winner_test,y_test))/len(winner_test)
    y_hat_test = classifier.predict(X_test)
    y_hat_train = classifier.predict(X_train)     
    

    if plot:
        if len(y) > max_plot_len:
            endi = math.floor(max_plot_len/2)
            # print(prob_test.shape)
            # print(prob.shape)
            y_plot=y_test[0:endi]
            winner_plot = winner_test[0:endi]
            prob_plot = prob_test[0:endi,:]
            y_hat_plot = y_hat_test[0:endi]
        else:
            prob = classifier.predict_proba(x)
            y_hat = classifier.predict(x)
            winner = get_wta_output(prob,input_T)
            wta_accuracy = np.sum(np.equal(winner,y))/len(winner)
            y_plot=y
            prob_plot = prob
            winner_plot=winner
            y_hat_plot = y_hat
        channels = prob_plot.shape[1]

        if downsample_rate:
            t = [n*downsample_rate for n in range(len(y_plot))]
        else:
            t = [n for n in range(len(y_plot))]


        # if 1:
        if plot == 1:
                    
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_yticks([])
            ax2.set_yticks([])
            f.set_size_inches(FIG_WIDTH, FIG_HEIGHT)        
            
            for i in range(channels):
                ax1.plot(t,np.add(y_plot==(i),i*1.1),color='k',linewidth=1)
                ax1.plot(t,np.add(prob_plot[:,i],i*1.1),linewidth=1)
            ax1.set_xlabel("timesteps")

            ax2.plot(t,y_plot,label='y',color = 'k')
            ax2.plot(t,winner_plot + 1.1*max(y),label='pred after WTA',color = COLOURS[3])
            ax2.plot(t,y_hat_plot+2*1.1*np.max(y),label='pred before WTA',color=COLOURS[4])

        elif plot == 2:
            prob_plot = prob_plot[:,1]
            f, axes = plt.subplots(4, 1)
            f.set_size_inches(FIG_WIDTH, FIG_HEIGHT)        
            
            for ax in axes[0:len(axes)-1]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[-1].set_yticks([])

            # predicted 
            axes[0].hlines(0.5,min(t),max(t),linestyle='dashed',color='k')
            axes[0].plot(t,prob_plot)
            axes[0].set_ylabel("W*x")

            # predicted before WTA
            axes[1].plot(t, y_hat_plot,label='pred b4 WTA',color = COLOURS[4])
            axes[1].set_ylabel("W*X Th")

            # predicted after WTA
            axes[2].plot(t, winner_plot,label='pred after WTA',color = COLOURS[3])
            axes[2].set_ylabel("Predicted")

            # true
            axes[3].plot(t,y_plot,label='y',color = 'k')
            axes[3].set_ylabel("True")

            axes[-1].set_xlabel("timesteps")
    
        plt.tight_layout()
        if plot < 3:
            plt.show()

    if confusion_matrix_:
        y_hat_test = classifier.predict(X_test)
        plot_confusion_matrix(y_test,y_hat_test,winner_test,input_T)
    train_instantaneous_sucess_rate = np.sum(np.equal(y_hat_train,y_train))/len(y_train)
    test_instantaneous_sucess_rate = np.sum(np.equal(y_hat_test,y_test))/len(y_test) 
    if plot:
        print(f'train accuracy instantaneous = {train_instantaneous_sucess_rate:.4f}')
        print(f'test accuracy instantaneous = {test_instantaneous_sucess_rate:.4f}')
        print(f'test accuracy wta = {wta_accuracy_test:.4f}')
        
    return [wta_accuracy_test, test_instantaneous_sucess_rate,train_instantaneous_sucess_rate], classifier, x,y 

def smooth(probs,input_T):
    winner=np.empty(len(probs))
    for start_index in range(0,len(probs),input_T):
        end_index = len(probs)
        if end_index > start_index + input_T:
            end_index = start_index+input_T
        averages=np.mean(probs[start_index:end_index],axis=0)
        result = np.where(averages > 0.5,1,0)
        winner[start_index:end_index] = result
    return winner

def do_linear_regression(compete_for_events,currents,y,tau,threshold,start_index,plot,input_T,extra_time_scales=0,type_='e', confusion_matrix_=0,max_plot_len=2000000,downsample_rate=0,bias=False):
    ''' type can be e - event train, d_i - delta I, abs_d_i - abs delta I'''
    
    # input T scaled by downsample rate
    if downsample_rate:
        if np.remainder(input_T,downsample_rate):
            raise Exception("input period must be divisible by downsample rate")
        else:
            input_T = int(input_T/downsample_rate)
            
    x = get_LIF_readout_from_current(currents,tau,threshold,start_index,extra_time_scales,type_)
    if bias:
        x=np.c_[x,np.ones(len(x))]    # bias
    # exclude init data
    if start_index != 0:
        y=y[start_index:len(y)-1]
        x=x[start_index:len(x),:]
    
    if downsample_rate!=0:
        x=np.roll(x,-10) # wait  a bit to start sampling
        x=x[::downsample_rate]
    
        y=np.roll(y,-10) # wait a bit to start sampling
        y=y[::downsample_rate]
        max_plot_len = np.rint(max_plot_len / downsample_rate)

    X_train,X_test,y_train,y_test=split_into_train_test(x,y)
    
  # Linear Regression
    x_pinv = np.linalg.pinv(X_train, rcond=1e-15)
    weights = np.matmul(x_pinv, y_train)
    prob_test =np.matmul(weights,X_test.T)
    prob_train =np.matmul(weights,X_train.T)

    winner_test = smooth(prob_test,input_T)
    winner_train = smooth(prob_train,input_T)

    wta_accuracy_test= np.sum(np.equal(winner_test,y_test))/len(winner_test)
    wta_accuracy_train= np.sum(np.equal(winner_train,y_train))/len(winner_test)

    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_yticks([])
        ax2.set_yticks([])
        f.set_size_inches(FIG_WIDTH, FIG_HEIGHT) 
        
        if len(y) > max_plot_len:
            endi = math.floor(max_plot_len/2)
            # print(prob_test.shape)
            # print(prob.shape)
            y_plot=y_test[0:endi]
            winner_plot = winner_test[0:endi]
            prob_plot = prob_test[0:endi]
            # y_hat_plot = y_hat_test[0:endi]
        else:
            prob = np.matmul(weights,x.T)
            # y_hat = classifier.predict(x)
            winner = smooth(prob,input_T)
            wta_accuracy = np.sum(np.equal(winner,y))/len(winner)
            y_plot=y
            prob_plot = prob
            winner_plot=winner
        
        ax1.plot(np.add(y_plot==(1),1.1),color='k',linewidth=1)
        ax1.plot(np.add(prob_plot,1.1),linewidth=1,color='r')
        ax1.set_title('predicted probabillity for each output class')
        # plt.title("Test data predicted probability")
        
        ax2.plot(y_plot,label='y')
        ax2.plot(winner_plot + 1.1*max(y),label='pred after WTA')
        # ax2.plot(y_hat_plot+2*1.1*np.max(y),label='pred before WTA')
        ax2.legend(loc=4)
        # ax2.set_title('y predicted - y after WTA - y before WTA')
        plt.tight_layout()
        plt.show()
        # plt.title("Test data predicted y vs y")
        
    if plot:
        # train_instantaneous_sucess_rate = np.sum(np.equal(y_hat_train,y_train))/len(y_train)
        # test_instantaneous_sucess_rate = np.sum(np.equal(y_hat_test,y_test))/len(y_test) 
        # print(f'train accuracy instantaneous = {train_instantaneous_sucess_rate:.4f}')
        # print(f'test accuracy instantaneous = {test_instantaneous_sucess_rate:.4f}')
        print(f'train accuracy wta = {wta_accuracy_train:.4f}')
        print(f'test accuracy wta = {wta_accuracy_test:.4f}')
        mse = mean_squared_error(prob_train,y_train)
        print(f'mean_squared_error = {mse:.4f}')

    return wta_accuracy_test, weights, x,y 



def do_logit(compete_for_events,currents,y,tau,threshold,start_index,plot,input_T,extra_time_scales=0,type_='e', confusion_matrix_=0,max_plot_len=2000000):
    '''
    logistic regression 
    type can be e - event train, d_i - delta I, abs_d_i - abs delta I
    '''
  
    x = get_LIF_readout_from_current(currents,tau,threshold,start_index,extra_time_scales,type_)

    # exclude init data
    y=y[start_index:len(y)-1]
    x=x[start_index:len(x),:]
    
    X_train,X_test,y_train,y_test=split_into_train_test(x,y)

    # Logistic regression classifier
    classifier = LogisticRegression(multi_class='multinomial').fit(X_train, y_train) # multinnomial makes it use cross entropy loss rather than 1 vs rest
    
    prob_test = classifier.predict_proba(X_test)
    winner_test = get_wta_output(prob_test,input_T)
    wta_accuracy_test = np.sum(np.equal(winner_test,y_test))/len(winner_test)

    if plot:
        plt.figure()
        if len(y) > max_plot_len:
            y_hat_test = classifier.predict(X_test)
            endi = math.floor(max_plot_len/2)
            # print(prob_test.shape)
            # print(prob.shape)
            y_plot=y_test[0:endi]
            winner_plot = winner_test[0:endi]
            prob_plot = prob_test[0:endi,:]
            y_hat_plot = y_hat_test[0:endi]
        else:
            prob = classifier.predict_proba(x)
            y_hat = classifier.predict(x)
            winner = get_wta_output(prob,input_T)
            wta_accuracy = np.sum(np.equal(winner,y))/len(winner)
            y_plot=y
            prob_plot = prob
            winner_plot=winner
            y_hat_plot = y_hat
        for i in range(prob_plot.shape[1]):
            plt.plot(np.add(y_plot==(i),i*1.1),'b')
            plt.plot(np.add(prob_plot[:,i],i*1.1),linewidth=0.5)
        plt.ylabel('predicted probabillity for each output class')
        plt.show()
        # plt.title("Test data predicted probability")
        
        plt.plot(y_plot)
        plt.plot(winner_plot + 1.1*max(y))
        plt.plot(y_hat_plot+2*1.1*np.max(y))
        plt.ylabel('y predicted - y after WTA - y before WTA')
        plt.show()
        # plt.title("Test data predicted y vs y")

    if confusion_matrix_:
        plot_confusion_matrix(y_test,y_hat_test,winner_test,input_T)

   
    if plot:
        y_hat_test = classifier.predict(X_test)
        y_hat_train = classifier.predict(X_train)         
        train_instantaneous_sucess_rate = np.sum(np.equal(y_hat_train,y_train))/len(y_train)
        test_instantaneous_sucess_rate = np.sum(np.equal(y_hat_test,y_test))/len(y_test) 
        print(f'train accuracy instantaneous = {train_instantaneous_sucess_rate:.4f}')
        print(f'test accuracy instantaneous = {test_instantaneous_sucess_rate:.4f}')
        print(f'test accuracy wta = {wta_accuracy_test:.4f}')
        
    return wta_accuracy_test, classifier, x,y 

def get_pairwise_accuracy(currents,voltages,input_T,operation='xor',prnt=True,prnt_train=False,debug=False,threshold=0.000001,downsample_rate=2000,tau=TAU,input_type=['average'],extra_xor=False,type_='e',input_mp=False,include_123=False,**kwargs):
    """
    prints and returns pairwise func accuracy
    func can be or, and, xor 
    if you want to overwrite default in kwargs with another function default, you have to pass it into the funciton when you call it.
    """
    if prnt:
        print()
        print(f"Pairwise accuracy, mode = {operation}")
    if operation == 'xor':
        func = np.logical_xor
    elif operation == 'or':
        func = np.logical_or
    elif operation == 'and':
        func = np.logical_and
    elif operation == 'nor' or operation=='nand':
        func = 'special'
    else:
        raise Exception("Mode not valid")

    if operation == 'nor':
        [v1,v2,v3] = voltages
        high = np.max(v1)

        if operation == 'nor':
            trues = [
            3,
            np.logical_not(np.logical_or(v1==high,v3==high)),
            np.logical_not(np.logical_or(v2==high,v3==high))
        ]
        labels = [
                f'v1 {operation} v2 accuracy = ',
                f'v1 {operation} v3 accuracy = ',
                f'v2 {operation} v3 accuracy = '
            ]
    else:
        if len(voltages) == 4:
            [v1,v2,v3,v4] = voltages
            high = np.max(v1)
            trues = [
                func(v1==high,v2==high),
                func(v1==high,v3==high),
                func(v1==high,v4==high),
                func(v2==high,v3==high),
                func(v2==high,v4==high),
                func(v3==high,v4==high)
            ]
            labels = [
            f'v1 {operation} v2 accuracy = ',
                f'v1 {operation} v3 accuracy = ',
                f'v1 {operation} v4 accuracy = ',
                f'v2 {operation} v3 accuracy = ',
                f'v2 {operation} v4 accuracy = ',
                f'v3 {operation} v4 accuracy = '
            ]
            
        elif len(voltages) == 3:
            [v1,v2,v3] = voltages
            high = np.max(v1)
            trues = [
                func(v2==high,v1==high),
                func(v1==high,v3==high),
                func(v2==high,v3==high)
            ]
            labels = [
                f'v1 {operation} v2 accuracy = ',
                f'v1 {operation} v3 accuracy = ',
                f'v2 {operation} v3 accuracy = '
            ]
            if include_123:
                labels.append('v1 {operation} v2 {operation} v3 accuracy = ')
                trues.append(func(v1==high,func(v2==high,v3==high)))
        elif len(voltages) == 2:
            [v1,v2] = voltages
            high = np.max(v1)
            trues = [
                func(v1==high,v2==high),
                # func(v1==high,v2==high,v3==high)
            ]
            labels = [
                f'v1 {operation} v2 accuracy = ',
            ]
        else:
            raise Exception("Can't deal with the number of input voltages.")
    a=[]

    if extra_xor:
        trues.append((v1==high)^(v2==high)^(v3==high))
        labels.append("\nExtra functions\nv1 xor v2 xor v3 accuracy = ")
        trues.append((v1==high)^(v2==high)^(v4==high))
        labels.append("v1 xor v2 xor v4 accuracy = ")
        trues.append((v2==high)^(v3==high)^(v4==high))
        labels.append("v2 xor v3 xor v4 accuracy = ")
        trues.append((v1==high)^(v3==high)^(v4==high))
        labels.append("v1 xor v3 xor v4 accuracy = ")
        trues.append((v1==high)^(v2==high)^(v3==high)^(v4==high))
        labels.append("v1 xor v2 xor v3 xor v4 accuracy = ")
        
        trues.append((v1==high))
        labels.append("v1 accuracy = ")
        trues.append((v2==high))
        labels.append("v2 accuracy = ")
        trues.append((v3==high))
        labels.append("v3 accuracy = ")
        trues.append((v4==high))
        labels.append("v4 accuracy = ")


        trues.append((v1==high)^(v2==high))
        labels.append("\nFull adder\nQ0 accuracy = ")
        trues.append(np.logical_xor(np.logical_xor(v3==high,v4==high),np.logical_and(v1==high,v2==high)))
        labels.append("Q1 accuracy = ")
        and_input = np.logical_and(np.logical_xor(v3==high,v4==high),np.logical_and(v1==high,v2==high))
        trues.append(np.logical_or(and_input, np.logical_and(v3==high,v4==high)))
        labels.append("CO accuracy = ")

    for true,label in zip(trues,labels):
#   def do_LDA(currents,y,tau,threshold,input_T,plot=PLOT,confusion_matrix_=PLOT,max_plot_len=2000000,downsample_rate=0,input_type=[''],**kwargs):
        [wta_accuracy_test, test_instantaneous_sucess_rate,train_instantaneous_sucess_rate],_,_,_=do_LDA(currents,true,tau,threshold,input_T,downsample_rate=downsample_rate,input_type=input_type,debug=debug,type_=type_,input_mp=input_mp,**kwargs)
        if prnt_train:
            print(f'train: {label} {train_instantaneous_sucess_rate:.2f}')
            print(f'test: {label} {wta_accuracy_test:.2f}')
        elif prnt:
            print(f'{label} {wta_accuracy_test:.2f}')


        a.append(wta_accuracy_test)
    return a 


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def display_accuracy(currents,voltages,input_T=10000,show=True,ops=['xor','and','or'],include_123=False,**kwargs):
    """ display accuracy as table
    pairwise accuracy for xor,or,and
    this version allows you to pass in currents, y and voltages ([v1,v2...])
     """


    a_ob = [] # shape [[xor pairwise],[or pariwise],[and_pairwise]]

    for op in ops:
        acc = get_pairwise_accuracy(currents,voltages,input_T,operation=op,**kwargs)
        acc = [round(a,2) for a in acc]
        a_ob.append(acc)

    df = pd.DataFrame(a_ob)


    if len(voltages)==4:
        labels = [
            "(1,2)",
            "(1,3)",
            "(1,4)",
            "(2,3)",
            "(2,4)",
            "(3,4)"
        ]
    elif len(voltages)==3:
        if include_123:
            labels = [
                "(1,2)",
                "(1,3)",
                "(2,3)",
                "(1,2,3)",
            ]
        else:
            labels = [
                "(1,2)",
                "(1,3)",
                "(2,3)"
            ]
    df = df.set_axis(labels,axis=1)
    df = df.set_axis(["XOR","AND","OR"])

    if show:
        make_table_for_pdf([df],show_row=True)
    return df



def event_rate_mapping(currents,y,voltages,by_electrode=True,pattern=-1,ax=[],legend=False,**kwargs):
    """ 
    plot relationship between er and current or voltage. 
    if by_electrode is true, colour code by electrode
    otherwise, colour code by pattern
    ax is list of axes [ax] 
    """
    ms=3
    edge_col='k'
    tau=0

    if ax == []:
        f,ax = plt.subplots(1)
        f.set_size_inches(7,5)
        pass_in_ax=False
    else:
        ax = ax[0]
        pass_in_ax=True

    x = get_LIF_readout_from_current(currents,tau,**kwargs)
    e=0
    # print(x.shape)
    done=0
    for pattern in range(8):

        # event_rates = (np.sum(x[y==pattern],axis=0))
        cs=downsample_flexible((currents[:,y==pattern]).T,100000)
        xs=downsample_flexible(x[y==pattern,:],100000)

        if by_electrode:
            e=0

            for i,k in zip(cs.T,xs.T):
                if not done:
                    ax.plot(i,k,'o',label='E' + str(e+1),color=COLOURS[e],alpha=TRANSPARENCY,markersize=ms,markeredgecolor=COLOURS[e])
                else:
                    ax.plot(i,k,'o',label='_nolegend_',color=COLOURS[e],alpha=TRANSPARENCY,markersize=ms,markeredgecolor=COLOURS[e])
                e+=1

        else: # by_pattern
            p=str((np.array(voltages)[:,y==pattern][:,0]==8).astype(int))
        
            for i,k in zip(cs.T,xs.T):

                if e==0:
                    ax.plot(i,k,'o',label=p,color=COLOURS[pattern],alpha=TRANSPARENCY,markersize=ms,markeredgecolor=COLOURS[e])
                else:
                    ax.plot(i,k,'o',label='_nolegend_',color=COLOURS[pattern],alpha=TRANSPARENCY,markersize=ms,markeredgecolor=COLOURS[e])
                e+=1
        done=1
    

        if legend:
            ax.legend()
    
    if not pass_in_ax:
        plt.xlabel('<Voltage>')
        plt.ylabel('Event Rate')
        plt.show()




def plot_accuracy_vs_threshold(currents,voltages,input_T,thresholds=[],**kwargs):
    if thresholds==[]:
        thresholds=[0.00001,0.001,0.01,0.1,0.2,0.5,1,1.5,2,3,5,6,7]

    accuracy_array = []
    voltages=np.array(voltages)

    for threshold in thresholds:
        a_ob = [] # shape [[xor pairwise],[or pariwise],[and_pairwise]]
        try:
            for op in ['xor','or','and']:
                a_ob.append(get_pairwise_accuracy(currents,voltages,input_T,operation=op,prnt=0,threshold=threshold,tau=0,**kwargs))
        except ValueError:
            a_ob=[0,0,0]
            a_ob = [a_ob for el in range(3)]
            print(f"Couldn't solve for threshold = {threshold}")
        accuracy_array.append(a_ob)

    a=np.array(accuracy_array)
    labels = [    
    'V1, V2',
    'V1, V3',
    'V2, V3'
    ]
    
    plot_xor_or_and(a,thresholds,'threshold','log',labels)
    return a


def plot_event_rate_mapping_by_electrode(voltages_out,y,size=[2,4],downsample_rate=50000,by_pattern=True,title = '',xlim=[],**kwargs):
    """
    if pass in title, save fig as png, otherwise call plt.show()
    """
    f, axes = plt.subplots(size[0],size[1])
    f.set_size_inches(8,4)
    axes = axes.flatten()
    x = get_LIF_readout_from_current(voltages_out,0,**kwargs)
    cs=downsample_flexible((voltages_out).T,downsample_rate)
    xs=downsample_flexible(x,downsample_rate)

    for pattern in range(8):

        # event_rates = (np.sum(x[y==pattern],axis=0))
        cs=downsample_flexible((voltages_out[:,y==pattern]).T,downsample_rate)
        xs=downsample_flexible(x[y==pattern,:],downsample_rate)
        
        # p=str((np.array(voltages)[:,y==pattern][:,0]==8).astype(int))
        e=0
        for i,k,ax in zip(cs.T,xs.T,axes):
            if by_pattern:
                ax.plot(i,k,'.',label='E' + str(e),color=COLOURS[pattern])
            else:
                ax.plot(i,k,'.',label='E' + str(e),color=COLOURS[e])
            e+=1
            ax.set_xlabel('E' + str(e))
            if len(xlim) == 2:
                ax.set_xlim(xlim)
            ax.set_yticks([])

    # ax.legend()
    plt.tight_layout()

    if title != '':
        plt.savefig(title + '.png',dpi=200)
    else:
        plt.show()


def which_cases_fail(currents,voltages,func=np.logical_xor,input_T=10000,downsample_rate=1000,**kwargs):
    """ 
    show which cases fail as a table for pairwise func (default XOR) 
    """
    [v1,v2,v3,v4] = voltages
    high=max(v1)
    if (max(v1)==min(v1)):
        raise Exception(" max v1 = min v1")
    trues = [
        func(v1==high,v2==high),
        func(v1==high,v3==high),
        func(v1==high,v4==high),
        func(v2==high,v3==high),
        func(v2==high,v4==high),
        func(v3==high,v4==high)
    ]
    labels = [
        "XOR(1,2)",
        "XOR(1,3)",
        "XOR(1,4)",
        "XOR(2,3)",
        "XOR(2,4)",
        "XOR(3,4)"
    ]
    patterns,l = get_patterns_as_int(voltages,return_lookup=True)
    df_pairwise_accuracies=pd.DataFrame()
    i=-1
    for true , label in zip(trues,labels):
        i +=1
        aargs=do_LDA(currents,true,input_T=input_T,downsample_rate=downsample_rate,**kwargs)
        classifier = aargs[1]
        x=aargs[2]
        y_hat = get_wta_output(classifier.predict_proba(x),int(input_T/downsample_rate))
        patterns = downsample_flexible(patterns,input_T,class_vector=1).astype(int)
        true = downsample_flexible(true,input_T)
        guess = downsample_flexible(y_hat,int(input_T/downsample_rate)).astype(int)
        d=get_sucess_rates_df(l,true,guess,patterns)
        df_pairwise_accuracies.insert(i, label, [round(a,2) for a in d["sucess rate"]], True)
    average_accuracy = np.sum(np.array(df_pairwise_accuracies[labels]),axis=1)/6
    average_accuracy = [round(a,2) for a in average_accuracy]
    df_pairwise_accuracies.insert(i+1,'average',average_accuracy)
    df_pairwise_accuracies.insert(0, "pattern", d["pattern"], True)

    make_table_for_pdf([df_pairwise_accuracies],hilight_bad= True)
    return patterns,l,df_pairwise_accuracies

##########################################################################################
########################## Plot functions ################################################
##########################################################################################


def make_table_for_pdf(dfs,show_row=False,hilight_bad=False):
    """ dfs = list of dataframes
    make a table b/c saving df.head() doesn't work very well
    show_row = show row labels
     """
    df_pairwise_accuracies = dfs[0]        
    ccolors = plt.cm.BuPu(np.full(len(df_pairwise_accuracies.columns), 0.1))
    fig ,axes= plt.subplots(len(dfs),1)
    for i,df_pairwise_accuracies in enumerate(dfs):
        if len(dfs)>1:
            a = axes[i]
        else:
            a=axes
        if show_row:
             the_table = a.table(cellText=df_pairwise_accuracies.values,colLabels=df_pairwise_accuracies.columns,loc='top',rowLabels=df_pairwise_accuracies.axes[0],colColours=ccolors,bbox=[0,0,1,1],rowColours=ccolors)         
        else:
            the_table = a.table(cellText=df_pairwise_accuracies.values,colLabels=df_pairwise_accuracies.columns,loc='top',colColours=ccolors,bbox=[0,0,1,1])
                 
        the_table.set_figure(fig)
        a.axis("tight")
        a.axis('off')
        # a.set_position([0,0,1,1],"original")
        # a.set_aspect(0.5)
       
    fig.set_size_inches(7,0.3*df_pairwise_accuracies.shape[0])
    # fig.tight_layout()
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.box(on=False)

    if hilight_bad:
        i = -1
        for el in df_pairwise_accuracies:
            i += 1
            if "(" in el:
                j = 1
                for accuracy in df_pairwise_accuracies[el]:
                    if accuracy < 0.5:
                        the_table[(j, i)].set_facecolor("#ffa590")
                    j+=1
    

    plt.show()

    
def plot_confusion_matrix(y_test,y_hat_test,winner_test,input_T):
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(FIG_WIDTH, FIG_HEIGHT)        

    cm=confusion_matrix(y_test,y_hat_test)
    cax1= ax1.matshow(cm,cmap='inferno')
    ax1.set_title("Before WTA - test dataset")
    ax1.set_xlabel('Predicted class')
    ax1.set_ylabel('True class')
    ax1.xaxis.set_ticks_position('bottom')

    cm=confusion_matrix(remove_duplicates(y_test,input_T),remove_duplicates(winner_test,input_T))
    cax2= ax2.matshow(cm,cmap='inferno')
    ax2.set_title("After WTA - test dataset")
    ax2.set_xlabel('Predicted class')
    ax2.set_ylabel('True class')
    ax2.xaxis.set_ticks_position('bottom')
 
    plt.tight_layout()
    plt.colorbar(cax1,ax=ax1)
    plt.colorbar(cax2,ax=ax2)

    plt.show()
    
    



def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def accuracy_v_tau(currents,y,threshold,input_T,taus='default',downsample_rate=0,ax=0):
    """compare test accuracy for different time constants for abs delta I and event_train """
    if taus == 'default':
        taus = [1000,5000,10000,15000,40000,50000]
    taus.sort()
    a = []
    a_1=[]

    for i in range(0,len(taus)):
        compete_for_events=1
        accuracy,_,_,_ = do_LDA(currents,y,taus[i],threshold,input_T,type_='e',downsample_rate=downsample_rate)
        a.append(accuracy)
        # compete_for_events=0
        # x = do_LDA(compete_for_events,currents,y,taus[i],threshold,start_index,plot,input_T,type_='e')
        # a_3.append(x[2])
        accuracy,_,_,_ = do_LDA(currents,y,taus[i],threshold,input_T,type_='abs_d_i',downsample_rate=downsample_rate)
        a_1.append(accuracy)
        # x = do_LDA(compete_for_events,currents,y,taus[i],threshold,start_index,plot,input_T,type_='d_i')
        # a_2.append(x[2])
        
    if ax==0:
        plt.plot(taus,a,label="E, compete")
        plt.plot(taus,a_1,color='green',label="abs(delta(I))")
        plt.ylabel('test accuracy')
        plt.xlabel('time constant')
        plt.legend(loc=4)
        plt.show()
    else:
        ax.plot(taus,a,'o-',label="E, compete")
        ax.plot(taus,a_1,'o-',color='green',label="abs(delta(I))")
        ax.set_ylabel('test accuracy')
        ax.set_xlabel('time constant')
        # ax.legend(loc=4)
    return taus, a, a_1

def accuracy_v_threshold_current(currents,y,tau,input_T,start_index=0,plot=0,ths='default',show_not_compete=0,downsample_rate=0,ax=0):
    """compare test accuracy for different thresholds """
    if ths == 'default':
        ths = [0.0000001,0.0001,0.001,0.01,0.02,0.05, 0.1, 0.2, 0.5, 1, 2]
    ths.sort()
        
    a = []
    a_2=[]

    for i in range(0,len(ths)):
        accuracy,_,_,_ = do_LDA(currents,y,tau,ths[i],input_T,type_='e',downsample_rate=downsample_rate)
        a.append(accuracy)
        if show_not_compete:
            accuracy,_,_,_ = do_LDA(0,currents,y,tau,ths[i],start_index,plot,input_T,downsample_rate=downsample_rate)
            a_2.append(accuracy)
            
  

    if ax==0:   
        plt.plot(ths,a,label="compete")
        if show_not_compete:
            plt.plot(ths,a_2,label="not compete")
        plt.ylabel('test accuracy - after WTA')
        plt.xlabel('theshold current')
        plt.xscale("log")
        plt.legend(loc=4)
        plt.show()

    else:
        ax.plot(ths,a,'o-',label="compete")
        if show_not_compete:
            plt.plot(ths,a_2,'o-',label="not compete")
        # ax.set_ylabel('test accuracy - after WTA')
        ax.set_xlabel('theshold current')
        ax.set_xscale("log")
        # ax.legend(loc=4)
    return ths,a,a_2

def accuracy_fig(currents,y,tau,threshold,input_T,start_index=0,plot=0,ths='default',taus='default',show_not_compete=0,downsample_rate=0):
    f, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
  
    f.set_size_inches(8.5, 4)
    accuracy_v_tau(currents,y,threshold,input_T,start_index=0,plot=0,taus=taus,downsample_rate=0,ax=ax1)
    accuracy_v_threshold_current(currents,y,tau,input_T,start_index=start_index,plot=plot,ths=ths,show_not_compete=show_not_compete,downsample_rate=downsample_rate,ax=ax2)
    plt.tight_layout()
    ax1.legend(loc=4)
    ax2.legend(loc=4)
    plt.show()

    
def plot_projection(x,true, dir_1,dir_2,legend=0,lookup=0,downsample_rate = 0,info=1,triangles=True,show_classification_outcome=False,f_ax=[]): 
    """ 
    x is array of shape (n_samples, n_features)
    true is vector of shape n_samples
    project x to 2 dimensional space defined by direciton (dir_1, dir_2)
    colour represents class
    dir_1 and dir_2 need to have len = n_features
    if classificaiton_outcome = True, hilight if classificaiton was correct or not. true is vector of ints 0-3
    0 - true positive
    1 - false postive 
    2 - true negative
    3 - false negative
    """
    if len(f_ax)==0:
        f = plt.figure()
        f.set_size_inches(FIG_WIDTH-2.5, FIG_HEIGHT)
        subplot=False
    else:
        subplot=True
        f,ax=f_ax
    # X array-like of shape (n_samples, n_features)

    if len(dir_1) != len(dir_2) or len(dir_2) != x.shape[1]:
        raise Exception("wrong shape for directions or x")

    x_reduced  = np.c_[((x@dir_1),x@dir_2)]
    
    if downsample_rate:
        x_reduced = downsample_flexible(x_reduced,downsample_rate)
        true = downsample_flexible(true,downsample_rate)

    for p in (np.unique(true)):
        label = lookup[p] if lookup else bool(p)

        if show_classification_outcome:
            lookup = {}
            lookup[0] = "True, classifier correct"
            lookup[1] = "False, classifier incorrect"
            lookup[2] = "False, classifier correct"
            lookup[3] = "True, classifier incorrect"

            if p == 0: #tp
                plt.scatter(x_reduced[true==p,0],
                            x_reduced[true==p,1],
                            marker='o', label="True, classifier correct",edgecolors='black',facecolor='darkorange',s=70,linewidths=1)
            if p == 1:
                plt.scatter(x_reduced[true==p,0],
                            x_reduced[true==p,1],
                            marker='o', label=label,edgecolors='red',facecolor='cornflowerblue',s=70,linewidths=2)
                
            if p == 2:
                plt.scatter(x_reduced[true==p,0],
                            x_reduced[true==p,1],
                            marker='o', label=label,edgecolors='black',facecolor='cornflowerblue',s=70,linewidths=1)
                
            if p == 3:
                plt.scatter(x_reduced[true==p,0],
                            x_reduced[true==p,1],
                            marker='o', label=label,edgecolors='red',facecolor='darkorange',s=70,linewidths=2)
        else:            
            marker = 'o' if p<4 or not triangles else '^'
            o = plt if not subplot else ax
            colours = [COLOURS[3],COLOURS[0]]
            if len(np.unique(true)) == 2:
                o.scatter(x_reduced[true==p,0],
                        x_reduced[true==p,1],
                        marker=marker, label=label,edgecolors='black',alpha=TRANSPARENCY,color=colours[p])
            else: 
                o.scatter(x_reduced[true==p,0],
                        x_reduced[true==p,1],
                        marker=marker, label=label,edgecolors='black',alpha=TRANSPARENCY)
            # if subplot:
                range_=abs(np.min(x_reduced[:,0])-np.max(x_reduced[:,0]))
                ax.set_xlim([np.min(x_reduced[:,0])-range_*0.1,np.max(x_reduced[:,0])+range_*0.1])
                range_=abs(np.min(x_reduced[:,1])-np.max(x_reduced[:,1]))
                ax.set_ylim([np.min(x_reduced[:,1])-range_*0.1,np.max(x_reduced[:,1])+range_*0.1])
                # ax.set_ylim([np.min(x_reduced[:,1]),np.max(x_reduced[:,1])])
    if legend:
        plt.legend()
        # plt.legend(loc='center left', bbox_to_anchor=(1.3, 1))
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')

    if not subplot:
        plt.show()
        plt.ylabel('Direction 2')
        plt.xlabel('Direction 1')

    if info:
        # ["%.2f" % member for member in theList]
        print(f'Direction 1: {[round(w,2) for w in dir_1]}')
        print(f'Direction 2: {[round(w,2) for w in dir_2]}')






    
def plot_classes_PCA(x,true, legend=0,lookup=0,downsample_rate = 0,info=1,triangles=True,f_ax = []): 
    """ 
    plot classes in true by thier two principle components as determined by PCA
    x is array of shape (n_samples, n_features)
    """
    f = plt.figure()
    f.set_size_inches(FIG_WIDTH-2.5, FIG_HEIGHT)
    
    pca = decomposition.PCA(n_components=2)
    # X array-like of shape (n_samples, n_features)
    # pca.fit(x)
    x_reduced  = pca.fit_transform(x)
    
    if downsample_rate:
        x_reduced = downsample_flexible(x_reduced,downsample_rate)
        true = downsample_flexible(true,downsample_rate)

    for p in (np.unique(true)):
        if lookup:
            label = lookup[p]
        else:
            label = bool(p)
        if p < 10 or not triangles:
            plt.scatter(x_reduced[true==p,0],
                        x_reduced[true==p,1],
                        marker='o', label=label,edgecolors='black')
        else:
            plt.scatter(x_reduced[true==p,0],
                    x_reduced[true==p,1],
                    marker='^', label=label,edgecolors='black')
    if legend:
        plt.legend()
        # plt.legend(loc='center left', bbox_to_anchor=(1.3, 1))
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.show()

    if info:
        # ["%.2f" % member for member in theList]
        print(f'First principle component direction: {[round(w,2) for w in pca.components_[0,:]]}\
        \nExplained variance ratio: {round(pca.explained_variance_ratio_[0],2)}')
        print(f'Second principle component direction: {[round(w,2) for w in pca.components_[1,:]]}\
        \nExplained variance ratio: {round(pca.explained_variance_ratio_[1],2)}')                

def plot_separation_boundary_LDA(currents,true,tau,threshold,start_index,input_T,info=1,downsample_rate=0,lookup=0,legend=0,patterns='dont show',type_='e'):
    """ 
    Plot LDA direction and descision boundary for binary classification problem
    y axis is first principle component
    if pass in patterns, dot is colour of pattern not true/false class
    """
    accuracy, classifier, x,y =do_LDA(currents,true,tau,threshold,input_T,downsample_rate=downsample_rate,type_=type_)
    predicted = (x@classifier.coef_.T)
    pca = decomposition.PCA(n_components=1)
    x_reduced  = pca.fit_transform(x)
    
    f = plt.figure()
    f.set_size_inches(FIG_WIDTH-2.5, FIG_HEIGHT)
    
    if patterns == 'dont show':
        for p in (np.unique(y)):
            if lookup:
                label = lookup[p]
            else:
                label = bool(p)
            plt.scatter(predicted[y==p],
                        x_reduced[y==p],
                        marker='o', label=label,edgecolors='black')
    else:
        patterns = downsample_flexible(y, downsample_rate)
        for p in (np.unique(patterns)):
            if lookup:
                label = lookup[p]
            else:
                label = p
            plt.scatter(predicted[patterns==p],
                        x_reduced[patterns==p],
                        marker='o', label=label,edgecolors='black')
            # x_ = np.array([np.min(X[:,0], axis=0), np.max(X[:,0], axis=0)])
            # y = classifier.intercept_
            
    plt.axvline((-1*classifier.intercept_),linestyle='dashed',color='k',label='descision boundary')
    plt.ylabel('PC1')
    plt.xlabel('LDA')
    if legend:
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.show()
    if info:
        print(f'First principle component direction: {[round(w,2) for w in pca.components_[0,:]]}\
        \nExplained variance ratio: {round(pca.explained_variance_ratio_[0],2)}')        
        print(f'LDA direction: {[round(w,2) for w in classifier.coef_[0,:]]}')
        print(f'Classificaiton accuracy: {(accuracy[0])}')


def plot_logical_ops(accuracy,x,axes,xlabel,legend=0,log_scale=False):
    """
    like plot xor or and but prettier plots and specifically for 3 input bits
    """
    labels = [    
    'V1, V2',
    'V1, V3',
    'V2, V3'
    ]
    transparency=0.8
    ms=5
    colours = [COLOURS[0],COLOURS[2],COLOURS[3]]
    v_true=x
    markers = ['^--','o--','D--']
    [ax1,ax2,ax3] = axes
    for i,l in enumerate(labels): # for each pairwise combo 
        color=colours[i]
        __xor = accuracy[:,0,i]
        __or = accuracy[:,1,i]
        __and = accuracy[:,2,i]
        label = l 
        marker = 'o--'
        # color=next(ax1._get_lines.prop_cycler)['color']
        ax1.plot(v_true,__xor,marker,color=color,label=label,alpha=transparency,markersize=ms,markeredgecolor='k')
        ax2.plot(v_true,__or,marker,color=color,label=label,alpha=transparency,markersize=ms,markeredgecolor='k')
        ax3.plot(v_true,__and,marker,color=color,label=label,alpha=transparency,markersize=ms,markeredgecolor='k')

    ax1.set_ylabel('Accuracy')
    for ax in axes:
        if log_scale:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        # ax.set_ylim([0.4,1.05])
    ax1.set_title('XOR')
    ax2.set_title('OR')
    ax3.set_title("AND")
    if legend:
        ax3.legend(loc=4)

def plot_xor_or_and(accuracy,v_true,x_label='$V_{true}$',xscale='linear',labels = [],set_min=False,transparency=TRANSPARENCY):
    """ 
    accuracy shape (voltages,operations (eg or,xor,and),pairwise combos)
    accuracy takes form [[[xor pairwise],[or pariwise],[and_pairwise]],[[]]] for all voltaes v_true
    """

    labels = [    
        'V1, V2',
        'V1, V3',
        'V2, V3'
                ]
        
    f, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True)
    axes = [ax1,ax2,ax3]
    f.set_size_inches(FIG_WIDTH, FIG_HEIGHT*0.67)
    for i,l in enumerate(labels): # for each pairwise combo 
        __xor = accuracy[:,0,i]
        __or = accuracy[:,1,i]
        __and = accuracy[:,2,i]
        color=next(ax1._get_lines.prop_cycler)['color']
        ax1.plot(v_true,__xor,'o-',color=color,label=l,alpha=transparency)
        ax2.plot(v_true,__or,'o-',color=color,label=l,alpha=transparency)
        ax3.plot(v_true,__and,'o-',color=color,label=l,alpha=transparency)

    ax1.plot(v_true,[0.75 for el in v_true],'--',color="k")
    ax1.set_ylabel('accuracy')
    for ax in axes:
        ax.set_xscale(xscale)
    if set_min:
        ax1.set_ylim([60,ax1.get_ylim()[1]])
    ax1.set_title('XOR')
    ax2.set_title('OR')
    ax3.set_title("AND")
    ax2.set_xlabel(x_label)

    ax3.legend()
    plt.show()



