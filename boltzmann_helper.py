
"""
Boltzmanm Machine Helper Funcs 
Dependencies: snn_helper, sofies_plot_funcitons, analysis
"""

# Lab data constants
LENGTH_60E_16M = 16000000
LENGTH_SHORT_1_6_M = 1600000
FACTORS_945 = [(35, 27), (45, 21), (63, 15), (105, 9), (135, 7), (189, 5), (315, 3)]

from snn_helper import *  
from sofies_plot_functions import *
from analysis import *
import sys,os,pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys,os
import math
import random
import sympy 

try:
    from sympy import IndexedBase,expand,Subs,Sum,lambdify
    from sympy.abc import n
    from sympy import latex
    from sympy import Abs
    from sympy import Add
except:
    print('Cant find sympy -- intf class wont work')
from scipy.optimize import curve_fit



# Int Factorisation Coupling Class 

class IntF:
    def __init__(self,p,q,F,scale=1,special = 0, calculate_coupling = True):
        self.p=p
        self.q=q
        self.F=F

        #########################
        # get energy function ###
        #########################
        X = IndexedBase('x')
        Y = IndexedBase('y')
        f1 = (Sum(2**n*X[n], (n, 0, p)))
        f2 = (Sum(2**n*Y[n], (n, 0, q)))
        if special==1:
            E = ((f1*f2 - F)**2)**1/2
        elif special == 2:
            E = ((f1*f2 - F)**2)**1/4
        elif special == 3:
            """ for the fractional powers was getting stuck when (f1*f2 - F) = 0
            must be a numerical issue so added 1 to ensure always positive
            """
            E = ((f1*f2 - F)**2 + 1)**1/8
        elif special == 4:
            E = ((f1*f2 - F)**2)**1/160
        elif special == 5:
            E = ((f1*f2 - F)**2)**4
        elif special == 6:
            from sympy import exp
            E = exp((f1*f2 - F)**2)
        elif special == 7:
            E = 1/((f1*f2 - F) + 1)**(10)
        elif special == 8:
            E = (Abs(f1*f2 - F))**(1/5)
        elif special == 9:
            E = sympy.log((f1*f2 - F)**2 + 1)
        else:
            E = (f1*f2 - F)**2

        E = expand(E)
        E = expand(E)
        E = Subs(E, (X[0],), (1,)).doit()
        E = Subs(E, (Y[0],), (1,)).doit()
        E = expand(E)

        print(E)
        

        # binary numbers -- x = x**2 = x**3 etc
        for i in range(p+1):
            for pwr in range(2,6):
                E = E.replace(pow(X[i],pwr),pow(X[i],1))
        for i in range(q+1):
            for pwr in range(2,6):
                E = E.replace(pow(Y[i],pwr),pow(Y[i],1))
        self.E = E

        ############################################
        # get coupling / parital derivatives  ######
        ############################################
        funcs = [] # functions are partial derivatives wrt pbit state. Inputs for : [x1,x2,x3,x4,x5,y1,y2,y3]
        if calculate_coupling:
            for i in range(p+q):
                if i < p:
                    sym = X[i + 1]
                else:
                    sym = Y[i - p + 1]
                funcs.append(lambdify([X,Y],-1*self.E.diff(sym)))
            self.C = funcs

        print(f'IntF F = {F}, P  = {p}, Q = {q}\nSpecial = {special}')

    def __str__(self):
        return f'Int Factorisaiton Network Info\nF = {F} p = {p}, q = {q}\nE = ' + str(self.E)
    
    def get_energy_func(self):
        return self.E
    
    def get_coupling(self,i=-1):
        # get partial derivative of energy func. If i = -1, return list with function for all pbits
        if i == -1:
            info = self.C
        else: 
            info = self.C[i]
        return info
    
    def get_energy_func_terms(self):
        expr = self.E
        return (Add.make_args(expr))
    
    def get_number_of_E_terms(self):
        return len(Add.make_args(self.E))



# Pbit class
class P_bit:
    def __init__(self,random=True,state=0,prob=0):
        self.random=random
        self.state=state
        self.prob=prob
    
    def update(self,input):
        if self.random:
           r = np.random.randint(100)/100 
        else:
           r = 0.5
        m = (sig(input)-r)
        self.prop=m
        self.state=m>0
        return m
    



######################################################################################
####################### sINGLE elECTRODE VERSION #####################################
######################################################################################
# threshold adaptive coupling


class pbit_1E:
    """ 
    Each electrode is 1 or more PBits
    control parameter is threshold voltage
    Properties
    electrode: electrode (number) to attach to
    VT5050: 5050 threshold voltage for the channel 
    state: 1 or 0, current state
    """
    def __init__(self,electrode,VT5050,vprev,state=-1,tau=100,spiking_neuron=False):
        """
        if spiking_neuron, pbit can recieve no input while state is 1. After tau timesteps, state must return to 0.
        if not spiking_neuron, pbit can stay in 1 state indefinitely.
        """
        if state == -1:
            state = np.random.randint(100) > 50

        self.state=state
        self.electrode=electrode
        self.VT5050 = VT5050
        self.vprev = vprev
        self.state_count = 0 
        self.tau = tau
        self.spiking_neuron=spiking_neuron

    def update(self,v,VT,compete_array_entry):
        """ 
        update pbit state. Take current output voltage v, compete array entry (1 if won event, 0 otherwise) and threshold voltage VT which is like the input to the pbit 
        """
        self.state_count += 1

        # check for spikes
        spiked = False
        check_for_spikes = False
        if not self.spiking_neuron:
            check_for_spikes = True
        elif self.state == 0:
            check_for_spikes = True
            
        if check_for_spikes:
            delta_v = abs(v-self.vprev) 
            # print(f'v = {v}, vprev = {self.v_prev} delta_v = {delta_v}')
            if compete_array_entry and (delta_v > VT):
                spiked = True
      
      
      
        # update state 
        if spiked:
            self.state = 1
            self.state_count = 0
        if self.state_count > self.tau:
            self.state = 0
        
        # set vprev
        self.vprev = v




class pbit_tau:
    """ 
    Each electrode is 1 or more PBits
    Control parameter is tau - time in 1 or 0 state
    Threshold is constant - do not deal with vs etc, only event trains
    Currently no competition for events implemented
    """
    def __init__(self,electrode,tau5050,shift,exponent,mp=-1):
        """
        mp - membrane potential
        state is 1 if mp > 0, else 0
        """
        if mp == -1:
            self.mp = (np.random.randint(100)-50)/50
        else:
            self.mp = mp
        self.state= self.mp > 0

        self.electrode=electrode
        self.tau5050 = tau5050
        self.c = (tau5050 + shift)** - exponent

    def update(self,event,tau):
        """ 
        update pbit state. 
        event is from event train
        tau is input 
        """
        if tau < 1: 
            self.mp = 0
        elif event:
            self.mp = 1
        else:
            self.mp = self.mp - (1/tau)
        self.state = self.mp > 0


class pbit_tau_v2:
    """ 
    Each electrode is 1 or more PBits
    Control parameter is tau - time in 1 or 0 state
     V2:
    if b >= 0: tau = tau5050 + b*alpha and spikes cause mp to transition to 1 state (mp = 1)
    if b < 0: tau = -1*(tau5050 + abs(b)*alpha) and spikes cause mp to transition to 0 state (mp = -1)

    spikes cause mp to transition to self.default state (-1 if b < 0, otherise 1)
    
    Each electrode is 1 or more PBits
    Control parameter is tau - time in 1 or 0 state
    Threshold is constant - do not deal with vs etc, only event trains
    Currently no competition for events implemented

    """
    def __init__(self,electrode,tau5050,mp=-1,alpha=200):
        """
        mp - membrane potential
        state is 1 if mp > 0, else 0
        """
        if mp == -1:
            self.mp = (np.random.randint(100)-50)/50
        else:
            self.mp = mp

        self.state= self.mp > 0
        self.electrode=electrode
        self.tau5050 = tau5050
        self.alpha=alpha

    def update(self,event,bias):
        """ 
        update pbit state. 
        event is from event train
        tau is input 
        """
        if event:
            self.mp = 1 if bias > 0 else -1
        else:
            if bias > 0:
                tau = self.tau5050 + bias * self.alpha
            else:
                tau = -1*(self.tau5050 + abs(bias) * self.alpha)
            self.mp = self.mp - (1/tau)
        self.state = self.mp > 0
        return self.state
    
class pbit_sim_tau:
    """ 
    Each electrode is 1 or more PBits
    Offline - data already exists 
    This class is for factorisation of integer F with number of bits in x, y = p+1,q+1

    2 pbit versions (see 2 classes above)
    1: pbit_tau uses exponential 
    """
    def __init__(self,events,tau5050s, path = "results/pbit/CHANGE_IT_TAU", alpha=100, shift = 6, exponent = 0.01,p=2,q=2,F=35,n_replicas = 1,mapping=[],energy_function_variation = -1, v2_pbits = True):
        """" 
        events has n_electrode channels (rows) shape is (n_electrodes, samples)
        tau5050s is a list with tau5050 for each channel
        path is a relative path in .../results to save the data 
        """
        self.events = events
        print('Tau is control parameter')
        print(f'Pbits are version {2 if v2_pbits else 1}\nAlpha={alpha}')

        # params for tau equation
        self.tau5050s = tau5050s
        self.shift = shift
        self.alpha=alpha
        self.exponent = exponent

        # coupling depends on number of bits and int to factorise F
        num_pbits = p+q
        self.num_pbits = num_pbits
        self.p = p
        self.q = q
        if energy_function_variation > 0:
            int_factorisation = IntF(p,q,F,special=energy_function_variation)
        else:
            int_factorisation = IntF(p,q,F)
        self.coupling = int_factorisation.get_coupling()

        self.v2_pbits = v2_pbits

        # replicas
        self.n_replicas = n_replicas
        num_electrodes = events.shape[0]
        max_num_replicas = math.factorial(num_electrodes) / math.factorial(num_electrodes-num_pbits)
        assert(max_num_replicas < n_replicas, 'Too many replicas for the number of electrodes and pbits')
        self.num_electrodes=num_electrodes


        # get pbit objects with channel_num (electrode the pbit is attached to), threshold(will set initially) and state(randomly initalised)
        pbits = np.empty((n_replicas,num_pbits),dtype=object) # list with n_replicas rows, list of pbits for each replica

        # mapping array mapping pbits in each repilca to an electrode, shape is (repilcas,electrodes)
        e = self.num_electrodes
        p = self.num_pbits

        if mapping == []:
            mapping = np.empty((n_replicas,e))
            seq = [-1 if i >= p else i for i in range(e)]
            for r in range(n_replicas):
                unique=False
                while unique==False:
                    unique=True
                    random.shuffle(seq)
                    for row in mapping:
                        if (np.sum(row == seq) == len(seq)):
                            unique=False
                mapping[r,:] = seq
            print('Electrode to p-bit mapping:')
            print(mapping)

        for r in range(n_replicas):
            for n in range(num_pbits):
                e = np.where(mapping[r]==n)[0][0]
                if v2_pbits:
                    pbits[r,n] = pbit_tau_v2(e,tau5050s[e],alpha=alpha)
                else:
                    pbits[r,n] = pbit_tau(e,tau5050s[e],shift,exponent)
        self.pbits = pbits


        assert len(tau5050s) == events.shape[0], 'Wrong number of tau5050s or data is wrong shape'

        # log of all states with shape (state, replicas, # timesteps)
        # self.state_log = np.empty((num_pbits, n_replicas, events.shape[1]),dtype=bool)  
        self.state_log = np.empty((self.p + self.q, n_replicas, events.shape[1]),dtype=bool)  

    def run_async(self,timesteps=-1):
        """
        timesteps = -1 -> use all of data
        async mode
        """

        if timesteps == -1:
            timesteps = self.events.shape[1] - 2
      
        for t in range(timesteps): 
            # need to update each element in states -> need mapping from channels to pbits
            for replica in range(self.n_replicas):
                state_of_replica = [p.state for p in self.pbits[replica,:]]
                self.state_log[:,replica,t]=state_of_replica

                X_ = [1 if i == 0 else state_of_replica[i-1] for i in range(self.p+1)]
                Y_ = [1 if i == 0 else state_of_replica[i+self.p-1] for i in range(self.q+1)]
              
                for pbit_num in range(self.num_pbits):
                    p = self.pbits[replica,pbit_num]
                    event = self.events[p.electrode,t]
                    f = self.coupling[pbit_num]    
                    bias = f(X_,Y_)
                    # bias = 0
                    if self.v2_pbits:
                        p.update(event,bias)
                    else:   
                        tau = map_b_to_tau(bias,p,self.alpha,self.exponent,self.shift)
                        p.update(event,tau)

            # for replica in range(self.n_replicas):
            #     for pbit_num in range(self.num_pbits):
            #         bias = self.coupling[pbit_num]([1 if i == 0 else self.pbits[replica,i-1].state for i in range(self.p+1)],[1 if i == 0 else self.pbits[replica,i+self.p-2].state for i in range(self.q+1)])
            #         self.pbits[replica,pbit_num].update(self.events[self.pbits[replica,pbit_num].electrode,t],bias)


class pbit_sim_1E:
    """ 
    Each electrode is 1 or more PBits
    Offline - data already exists 
    This class is for factorisation of integer F with number of bits in x, y = p+1,q+1
    dependencies: boltzmann_helper
    """
    def __init__(self,data,V5050s,tau = 100,path = "results/pbit/CHANGE_IT_TAU", alpha=1,p=2,q=2,F=35,coupled=True,spiking_neuron=False,n_replicas = 1,bias_scale='lin',mapping=[]):
        """" 
        data has n_electrode channels. Data shape is (n_electrodes, samples) 
        V5050s is a list with V5050 for each channel
        path is a relative path in .../results to save the data 
        if spiking_neuron, pbit can recieve no input while state is 1. After tau timesteps, state must return to 0.
        if not spiking_neuron, pbit can stay in 1 state indefinitely.
        """


        self.events_no_th = event_train_array(data,0,compete=1,compete_mode='relative',threshold_array=[],debug_event_train_array=False,threshold_upper=-1)
        self.data=data
        self.bias = bias_scale

        # coupling depends on number of bits and int to factorise F
        num_pbits = p+q
        self.num_pbits = num_pbits
        self.p = p
        self.q = q
        int_factorisation = IntF(p,q,F)
        self.coupling = int_factorisation.get_coupling()


        # replicas
        self.n_replicas = n_replicas
        num_electrodes = data.shape[0]
        max_num_replicas = math.factorial(num_electrodes) / math.factorial(num_electrodes-num_pbits)
        assert(max_num_replicas < n_replicas, 'Too many replicas for the number of electrodes and pbits')
        self.num_electrodes=num_electrodes


        # randomly initialise state
        self.state=np.array([np.random.randint(100) > 50 for i in range(num_pbits)]) 
        self.state_count = np.array([0 for i in range(num_pbits)])


        # get pbit objects with channel_num (electrode the pbit is attached to), threshold(will set initially) and state(randomly initalised)
        pbits = np.empty((n_replicas,num_pbits),dtype=object) # list with n_replicas rows, list of pbits for each replica

        # mapping array mapping pbits in each repilca to an electrode, shape is (repilcas,electrodes)
        e = self.num_electrodes
        p = self.num_pbits

        if mapping == []:
            mapping = np.empty((n_replicas,e))
            seq = [-1 if i >= p else i for i in range(e)]
            for r in range(n_replicas):
                unique=False
                while unique==False:
                    unique=True
                    random.shuffle(seq)
                    for row in mapping:
                        if (np.sum(row == seq) == len(seq)):
                            unique=False
                mapping[r,:] = seq
            print('Electrode to p-bit mapping:')
            print(mapping)

        for r in range(n_replicas):
            for n in range(num_pbits):
                e,= np.where(mapping[r]==n)[0]
                # e=n
                pbits[r,n] = pbit_1E(e, V5050s[e], data[e,0],tau=tau,spiking_neuron=spiking_neuron)
        self.pbits = pbits

        # params 
        self.alpha=alpha
        self.tau = tau
        self.V5050s = V5050s
        self.coupled = coupled

        assert len(V5050s) == data.shape[0], 'Wrong number of threshold voltages or data is wrong shape'

        # log of all states with shape (state, replicas, # timesteps)
        self.state_log = np.empty((num_pbits, n_replicas, data.shape[1]))  

    def run_async(self,timesteps=-1):
        """
        timesteps = -1 -> use all of data
        async mode
        instead of applying V_in to each PNN sequentially, apply V_in to every PNN each timestep. 
        If PNN p1 spiked in the last tau timesteps then the state of x1 is 1, otherwise 0
        """

        if timesteps == -1:
            timesteps = self.data.shape[1] - 2
        for t in range(timesteps): 
            # need to update each element in states -> need mapping from channels to pbits
            for replica in range(self.n_replicas):
                state_of_replica = [p.state for p in self.pbits[replica,:]]
                X_ = [1 if i == 0 else state_of_replica[i-1] for i in range(self.p+1)]
                Y_ = [1 if i == 0 else state_of_replica[i+self.p-1] for i in range(self.q+1)]
                for pbit_num in range(self.num_pbits):
                    p = self.pbits[replica,pbit_num]
                    v = self.data[p.electrode,t+1]
                    compete_array_entry = self.events_no_th[p.electrode,t]
                    f = self.coupling[pbit_num]    
                    b = f(X_,Y_)
                    if self.coupled:
                        if self.bias== 'log':
                            VT = p.VT5050*10**(-1*b*self.alpha)
                        elif self.bias == 'lin':
                            VT = p.VT5050 -b*self.alpha
                            # if VT < 0:
                            #     VT=0
                        else:
                            raise Exception('invalid bias scale')

                    else:
                        VT = p.VT5050
                    p.update(v,VT,compete_array_entry)


            # str_ = ''
            # str_ += str(t)

            # for replica in range(self.n_replicas):
            #     for pbit_num in (range(self.num_pbits)):
            #         str_+= ','
            #         str_ += str(int(self.pbits[replica,pbit_num].state))
            # # str_+='\n'
            # self.pbit_file.write(str_)
            state = []
            for replica in range(self.n_replicas):
                state_rep = []
                for pbit_num in (range(self.num_pbits)):
                    state_rep.append(int(self.pbits[replica,pbit_num].state))
                self.state_log[:,replica,t]=state_rep


#############################################################################################
################## pbit sim tau helper func #################################################
#############################################################################################
def map_b_to_tau(b,p,alpha,exponent,shift):
    if p.c - b*alpha <= 0.1:
        tau = 1e6
    else:
        tau = (-1*(b * alpha) + p.c)**(-1/exponent) - shift

    if (p.c - b*alpha) <= 0:
        tau = 1e6
    else:
        tau = (-1*(b * alpha) + p.c)**(-1/exponent) - shift
        if tau > 1e6:
            tau = 1e6
    return tau



########################################################################################
############# DKL ANALYSIS #############################################################
########################################################################################

def kl_divergence(d1, d2): 
    """ 
    get KL divergence between two distributions 
    """   
    d1 = d1.flatten()
    d2 = d2.flatten()
    sum_=0
    for a,b in zip(d1,d2):
        if a != 0 and b!=0:
            sum_ += a * np.log(a / b)
    return sum_ 


def log_transform(im):
    '''returns log(image) scaled to the interval [0,1]'''
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
    except:
        pass
    return im


##########################################################################################################################################################################################################################################################################################

def get_state_from_current(currents,tau,threshold,events=[]):
    """
    return state vector x
    state is 1 if spiked (delta_I exceeds threshold) in the last tau timesteps, otherwise 0
    can also pass in an event train in which case currents and threshold will be ignored
    """
    if len(events) == 0:
        events = simple_event_train(currents,threshold)
    x = np.zeros(len(currents))
    count = tau
    for j,el in enumerate(events):
        if el == 1:
            count = 0
        if count < tau:
            x[j] = 1
        count += 1
    return x



def get_state_from_current_2(currents,tau,threshold,events=[]):
    """
    return state vector x
    state is 1 if spiked (delta_I exceeds threshold) in the last tau timesteps, otherwise 0
    can also pass in an event train in which case currents and threshold will be ignored
    faster? but slightly different...?
    """
    if len(events) == 0:
        events = simple_event_train(currents,threshold)
    kernel = np.ones((tau))
    x = np.convolve(events,kernel)>1
    return x

def sig(x):
 return 1/(1 + np.exp(-x))

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


def get_i5050(y):
    """ returns the index where y is closest to i50/50"""
    best = 0
    i=-1
    for idx,el in enumerate(y):
        if abs(el - 0.5) < abs(best-0.5):
            best = el
            i = idx
    return i




def get_threshold(abs_dvs, target_er= 0.02, learning_rate = 0.01 ,guess_threshold= 0.001, lim_guesses = 100,tol = 0.001, verbose=False):
    """ use grad descent to get threshold that gives target event rate target_er """
    converged = False
    guesses = 0
    while not converged and guesses < lim_guesses:
        events = simple_event_train_2(abs_dvs,guess_threshold)
        event_rate = np.sum(events) / len(events)
        error = event_rate - target_er
        if verbose:
            print(f'guess {guess_threshold}, event rate {event_rate}, error {error}')
        if abs(error) < tol:
            converged = True
            print(f'Error={error} - Converged')
            return (guess_threshold, event_rate)
        
        guess_threshold = guess_threshold + (error * learning_rate)
        guesses += 1
    if verbose: print('Did not converge')
    events = simple_event_train_2(abs_dvs,guess_threshold)
    event_rate = np.sum(events) / len(events)
    error = event_rate - target_er
    if verbose:
        print(f'guess {guess_threshold}, event rate {event_rate}, error {error}')
    return (guess_threshold, event_rate)


def get_approx_state_from_current(currents,tau,threshold,events=[]):
    """
    FASTEST and good agreement with get_state_from_current
    return approx state vector x
    this is faster but only approx - run with get_state_from_current_2 to be sure
    state is 1 if spiked (delta_I exceeds threshold) in the last tau timesteps, otherwise 0
    can also pass in an event train in which case currents and threshold will be ignored
    faster? but slightly different...?
    """
    if tau < 1: 
        return np.zeros(1)
    if len(events) == 0:
        events = simple_event_train(currents,threshold)
    # print(tau)
    events = events[:((len(events) // tau) * tau)].reshape(-1,tau)
    x = np.sum(events,axis=1) >= 1
   
    return x


def get_tau_5050(events,learning_rate = 1000,guess_tau=100, lim_guesses = 100,verbose=False ):
    """ use grad descent to get tau5050 """
    converged = False
    guesses = 0
    conv=0
    while not converged and guesses < lim_guesses:
        x = get_approx_state_from_current(None,guess_tau,0,events = events)
        exstate = np.sum(x) / len(x)
        error = exstate - 0.5
        if verbose:
            print(f'guess {guess_tau}, exstate {exstate}')
        new_guess_tau = round(guess_tau - (error * learning_rate))
        guesses += 1
        if new_guess_tau - guess_tau == 0:
            conv += 1
        else: 
            conv = 0
        if conv == 10:
            converged = True
        guess_tau = new_guess_tau
        if abs(error) < 0.005:
            converged=True
    return (guess_tau, exstate)


def check_correct(sd):
    """
    return true if the 2 max elements in sd correspond to 7,5 and 5,7 and they are similar densities (difference is less than half lagest)
    """
    x=list(sd.reshape(-1))
    x.sort()
    sums = x[-1]+x[-2]
    sums2 = sd[2,3] + sd[3,2]
    difference = abs( x[-1]-x[-2])
    return (sums == sums2) and  (difference < 0.5*max([x[-1]+x[-2]]))

def get_factors(states,i,p=2,q=2):
    """"
    shape of states is p+q, iterations
    return a list len i of factor pairs
    """
    sd = get_xy_density(states,p,q)
    x = sd.reshape(-1)
    x.sort()
    sd = get_xy_density(states,p,q)
    indices = np.where(sd >= x[-i])
    factor_pairs = []

    for x_ind,y_ind in zip(indices[0],indices[1]):
        x=x_ind*2+1
        y=y_ind*2+1
        factor_pairs.append((x,y))
    return factor_pairs

def print_factors(states,i,p,q,F,state_density=[],supress=False):

    if len(state_density)==0:
        sd = get_xy_density(states,p,q)
    else:
        sd = state_density
    sd_2=sd.copy()
    x = sd_2.reshape(-1)
    x.sort()
    min_ = x[-i]
    indices = np.where(sd >= min_ )

    factor_pairs = []
    counts = []
    for x_ind,y_ind in zip(indices[0],indices[1]):
        x=x_ind*2+1
        y=y_ind*2+1
        count = sd[x_ind,y_ind]
        counts.append(count)
        factor_pairs.append((count,x,y))
    
    if not supress:
        print(f'largest {i} (x,y) pairs are:')
    for f in sorted(factor_pairs,reverse=True):
        if f[0] > 0:
            if not supress:
                # print(f'x,y = {f[1]},{f[2]} ({f[2]*f[1]}), p = {f[0]:.3}, energy = {np.log((f[1]*f[2]-F)**2+1):.3}')
                print(f'x,y = {f[1]},{f[2]} ({f[2]*f[1]}), p = {f[0]:.3}, energy = {(f[1]*f[2]-F)**2}')

    if len(factor_pairs) > i:
        filtered_factor_pairs = []
        for idx, f in enumerate(sorted(factor_pairs,reverse=True)):
            if f[0] > 0:
                filtered_factor_pairs.append(f)
                break
        while(len(filtered_factor_pairs)) < i:
            filtered_factor_pairs.append(sorted(factor_pairs,reverse=True)[idx])
            idx += 1
        factor_pairs = filtered_factor_pairs
        
    return factor_pairs


def get_int_from_binary(states,p=2,q=2):    
    """ 
    states shape is pbits, replicas, iterations
    Convert binary values to ints
    Returns array of shape 2, iterations
    """
    # states = states.T.reshape(p+q,-1)
    print(f'shape is {states.shape}')
    if p == 8: # special case for P = 8, Q = 4, F = 945
        vals = np.zeros((8,states.shape[1]),dtype=bool)
        vals[-1,:] = True
        XS = vals.copy()
        for i in range(6):
            XS[6-i,:] = states[i,:]
        YS = vals.copy()
        for i in range(q):
            YS[6-i,:] = states[i+p,:]
        X = np.packbits(XS,axis=0) + states[7,:] * 2**8 + states[6,:] * 2 **7
        Y = np.packbits(YS,axis=0)
    else:
        vals = np.zeros((8,states.shape[1]),dtype=bool)
        vals[-1,:] = True
        XS = vals.copy()
        for i in range(p):
            XS[6-i,:] = states[i,:]
        YS = vals.copy()

        for i in range(q):
            YS[6-i,:] = states[i+p,:]
        X = np.packbits(XS,axis=0)
        Y = np.packbits(YS,axis=0)
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    X = X.astype(int)
    Y = Y.astype(int)
    return (X,Y)

def get_xy_density(states,p=2,q=2):
    """ 
    get a matrix of counts in each state corresponding 
    """
  
    X, Y = get_int_from_binary(states,p,q)
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    x_max = 2**(p+1) - 1
    y_max = 2**(q+1) - 1
    xs = [x for x in range(1,x_max+1,2)]
    ys = [y for y in range(1,y_max+1,2)]
    xy_density = np.zeros((len(xs),len(ys)))
    for i,x_ in enumerate(xs):
        for j,y_ in enumerate(ys):
            c = np.sum(np.logical_and(X==x_,Y==y_))
            xy_density[i,j] = c
    return xy_density / np.sum(xy_density)




def get_kl_div(sds,ref_dist):
    """
    sds object is array of states. Shape (timesections, replicas, sd1, sd2)
    state densities are over densities over time period in sim
    return kl divergence for timestep
    1) obtain sd as function of time
    2) compute kl divergence  

    """
    replicas = sds.shape[1]
    time_points = sds.shape[0]
    kl = np.empty((replicas,time_points))
    for r in range(replicas):
        for i in range(time_points):
            sds_small = sds[:i+1,r,:,:]
            sds_small = np.mean(sds_small,axis=0)
            kl[r,i] = kl_divergence(sds_small,ref_dist)
    return kl



def get_energies_and_densities(xy_densities, n = 7, p =8, q = 4, F = 945):
    """ 
    returns energy and density of most common n states given xy density arrays for various alphas.
    shape of xy_densities is: (len(alphas),x_size,y_size)
    """
    num_alphas = xy_densities.shape[0]
    energies = np.empty((n,num_alphas))
    densities = np.empty((n,num_alphas))

    for i in range(num_alphas):
        sd = xy_densities[i,:,:]
        factors = print_factors(None,n,p,q,F,state_density=sd,supress=True)
        for j, (density, X, Y) in enumerate(factors):
            energies[j,i] = np.log((X*Y-F)**2+1)
            densities[j,i]=density
    return (energies,densities)



##########################################################################
################# 0lder, slower fincs 
##########################################################################

# def get_int_from_binary(states,p=2,q=2):    
#     """ 
#     states shape is interations, pbits
#     for factorising 35. States shape is iterations, 4 
#     Convert binary values to ints
#     Returns array of shape 2, iterations
#     """
#     X = np.sum(states[:p,:].T*np.array([2**i for i in range(1,p+1)]),axis=1) + 1
#     Y = np.sum(states[p:,:].T*np.array([2**i for i in range(1,q+1)]),axis=1) + 1
#     return np.array([X,Y])

# def get_xy_density(states,p=2,q=2):
#     """ 
#     states shape is iterations, pbits
#     get a matrix of counts in each state corresponding 
#     this takes ages if you have lots of pbits.
#     """
#     states_int = get_int_from_binary(states,p,q)
#     x_max = 2**(p+1) - 1
#     y_max = 2**(q+1) - 1
#     xs = [x for x in range(1,x_max+1,2)]
#     ys = [y for y in range(1,y_max+1,2)]
#     xy_density = np.zeros((len(xs),len(ys)))
#     for i,x_ in enumerate(xs):
#         for j,y_ in enumerate(ys):
#             c = sum(np.where(states_int[0]==x_,1,0) + np.where(states_int[1]==y_,1,0) ==2)
#             xy_density[i,j] = c
#     return xy_density / np.sum(xy_density)


####################################################################
#################### PLOTTING FUNCS ################################
####################################################################


def plot_state_density_info(state_density,p=2,q=2,bits=False):
    """
    if bits = False, only plot state density for X and Y values
    if bits = True, plts state density of each pbit and probability of 0 and 1
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    fig.set_size_inches(9,3.5)     

   
    ax2 = fig.add_subplot(122)
    plot_2d_xy_density(None,p,q,[ax2],state_density=state_density)
    for (i, j), z in np.ndenumerate(state_density.T):
        ax2.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')


    plot_3d_xy_density(None,p,q,[ax],state_density=state_density)
    plt.show()


def plot_3d_binary_density(states,p=2,q=2,bits=False,ax=[]):
    """
    ax should be 3D if passing in
    """
    if len(ax) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.set_size_inches(7,7)   
    else: 
        ax=ax[0]

    # states=states.T
    density = np.empty((p+q,2))
    density[:,0] = np.sum(states==0,axis=0)
    density[:,1] = np.sum(states==1,axis=0)
    matrix = density.T # Replace with your 2D matrix data
    intensity = matrix  # Use the matrix values as intensity
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    X, Y = np.meshgrid(x, y)
    heights = intensity.flatten()
    x_pos, y_pos = np.meshgrid(x, y)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(heights)
    ax.bar3d(x_pos, y_pos, z_pos, dx=1, dy=1, dz=heights, shade=True, color=plt.cm.viridis(heights / heights.max()))
    ax.set_xlabel('P-bit')
    ax.set_ylabel('State of P-bit')
    ax.set_zlabel('Count')
    ax.set_xticks([0.5 + j for j in range(p+q)])
    # ax.set_xticklabels(['x1\np1','x2\np2','y1\np3','y2\np4'])
    ax.set_yticks([i+0.5 for i in range(2)])
    ax.set_yticklabels(['0','1'])


def plot_3d_xy_density(states,p=2,q=2,ax = [],state_density=[]):
    """
    if pass in ax, should be 3D
    """
    # Create a 3D plot
    if len(ax) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        fig.set_size_inches(7,7)   
    else: 
        ax=ax[0]

    if len(state_density)==0:
        xy_density = get_xy_density(states.T,p,q)
    else:
        xy_density = state_density
    matrix = xy_density.T # Replace with your 2D matrix data
    intensity = matrix  # Use the matrix values as intensity
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    heights = intensity.flatten()
    x_pos, y_pos = np.meshgrid(x, y)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(heights)
    ax.bar3d(x_pos, y_pos, z_pos, dx=1, dy=1, dz=heights, shade=True, color=plt.cm.viridis((heights - heights.min()) / (heights.max() - heights.min() )))


    # ax.bar3d(x_pos, y_pos, z_pos, dx=1, dy=1, dz=heights, shade=True, color=plt.cm.viridis(heights / heights.max() ))

    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel('P')
    ax.set_xticks([x + 0.5 for x in range(2**p)])
    ax.set_xticklabels([str(x) for x in range(1,2**(p+1),2)])
    ax.set_yticks([x + 0.5 for x in range(2**q)])
    ax.set_yticklabels([str(y) for y in range(1,2**(q+1),2)])
    ax.set_box_aspect((np.ptp(x_pos), np.ptp(y_pos), np.ptp((y_pos+x_pos)/4)))  # aspect ratio is 1:1:1 in data space


def plot_2d_xy_density(states,p=2,q=2,ax = [],state_density=[]):
    """
    
    """
    # Create a 3D plot
    if len(ax) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(9,3.5)   
    else: 
        ax=ax[0]

    
    if len(state_density)==0:
        xy_density = get_xy_density(states,p,q)
    else:
        xy_density = state_density

    ax.matshow(xy_density.T,cmap='viridis',interpolation='None')
    ax.set_xticks([x for x in range(2**p)])
    ax.set_xticklabels([str(x) for x in range(1,2**(p+1),2)])
    ax.set_yticks([x for x in range(2**q)])
    ax.set_yticklabels([str(y) for y in range(1,2**(q+1),2)])
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')



def plot_energy(states,p,q,F,ax=[],Ts=[],color='k',tau=1000,**kwargs):
     # Create a 3D plot
    if len(ax) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(7,3)   
    else: 
        ax=ax[0]
    states_int = get_int_from_binary(states,p,q)
    X = states_int[0,:]
    Y = states_int[1,:]
    energy_log = ((X*Y - F)**2)
    ax.plot([i/tau for i in range(len(energy_log))],energy_log,color=color,**kwargs)
    # ax.set_yscale('log')
    if len(Ts) > 0:    
        ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.25)
        ax2 = ax.twinx()
        ax2.plot([i/tau for i in range(len(energy_log))],Ts,'--',color='r')
        ax2.set_ylabel(r'Temperature (∝$\alpha$)')
        ax2.set_ylim([-3,1.1])
        ax2.set_yticks([0,0.5,1])
    ax.set_xlabel(R't ($\tau$)')
    ax.set_ylabel(r'Energy = $(XY-F)^2$')
    ax.set_ylabel(r'Energy')



def plot_energy_density(states,p,q,F,ax=[],Ts=[],k=-1,color='k'):
     # Create a 3D plot
    if len(ax) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(3,3)   
    else: 
        ax=ax[0]
    states_int = get_int_from_binary(states,p,q)
    X = states_int[0,:]
    Y = states_int[1,:]
    energy_log = ((X*Y - F)**2)
    if k == -1:
        k=50
    ax.hist(energy_log,orientation='horizontal',bins=[i*k for i in range(int(max(energy_log)/k))],color=color)
    # ax.set_yscale('log')
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel(r'Energy = $(XY-F)^2$')

def plot_energy_change_density(states,p,q,F,ax=[],Ts=[],k=-1,color='k',**kwargs):
     # Create a 3D plot
    if len(ax) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(3,3)   
    else: 
        ax=ax[0]
    states_int = get_int_from_binary(states,p,q)
    X = states_int[0,:]
    Y = states_int[1,:]
    energy_log = ((X*Y - F)**2)
    diff = np.diff(energy_log)
    change_energy_log= np.delete(diff,diff==0)
    # change_energy_log=diff
    if k ==-1:
        k=100
    ax.hist(change_energy_log,orientation='vertical',bins=[i*k - max(change_energy_log) for i in range(int(max(change_energy_log)*2/k))],color=color,**kwargs)
    # ax.set_yscale('log')
    
    ax.set_ylabel('Frequency')
    ax.set_xlabel(r'Energy difference')




def plot_sd_945(sd,filename='',p=5,q=5,factors_to_print=True):
    """ 
       save as 'filename'
    """

    av = sd
    if factors_to_print:
        print_factors(None,10,p,q,945,sd)

    fig = plt.figure()
    ax = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122)

    fig.set_size_inches(7,2.5)   

    xy_density = av
    matrix = xy_density.T # Replace with your 2D matrix data
    intensity = matrix  # Use the matrix values as intensity
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    heights = intensity.flatten()
    x_pos, y_pos = np.meshgrid(x, y)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(heights)
    # ax.view_init(30, 60)
    ax.bar3d(x_pos, y_pos, z_pos, dx=1, dy=1, dz=heights, shade=True, color=plt.cm.viridis((heights - heights.min()) / (heights.max() - heights.min() )))

    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel('P')
    ax.set_box_aspect((np.ptp(x_pos), np.ptp(y_pos), np.ptp((y_pos+x_pos)/3)))  # aspect ratio is 1:1:1 in data space

    ax2.matshow(xy_density,cmap='viridis',interpolation='none')
    gap=3
    for ax in [ax,ax2]:
        ax.set_xticks([x for x in range(0,2**p,2*gap)])
        ax.set_xticklabels([str(x) for x in range(1,2**(p+1),4*gap)])
        ax.set_yticks([x for x in range(0,2**q,2*gap)])
        ax.set_yticklabels([str(y) for y in range(1,2**(q+1),4*gap)])

    ax2.set_aspect(1)
    if len(filename) > 0:
        save_(filename)


def plot_sd_945_2d_pq84(density,ax):
    """ special 2d func for all factors of 945 """
    ax.matshow(density.T,aspect='auto',cmap='viridis',interpolation='None')
    ytick_names = np.array([3,5,7,9,15,21,27])
    xtick_names = np.array([35,45,63,105,135,189,315])
    yticks = (ytick_names - 1)/2
    xticks = (xtick_names - 1)/2
    ax.tick_params(axis="both",direction="out", left="off",bottom='off',labelleft="on")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_names)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_names)
    ax.set_ylabel(r'$Y$')
    ax.set_xlabel(r'$X$')
    ax.xaxis.set_ticks_position('bottom')

def add_red_box(ax,factor, sd, scale_linewidth = False):
    """ draw red box around factor """
    bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    display_aspect_ratio = bbox.width / bbox.height
    aspect = sd.shape[0]/sd.shape[1]/display_aspect_ratio
    if scale_linewidth:
        l = 400 / sd.shape[0]
    else:
        l = 0.5
    rect = plt.Rectangle((factor[0] // 2 - 0.75*aspect, factor[1] // 2 - 0.75),1.5*aspect, 1.5 , edgecolor='red', facecolor='none', linewidth= l)
    ax.add_patch(rect)

################################################################################################################
###################################### Funcs to deal with cpp sim results ######################################
################################################################################################################

def read_in_cpp_results(filename='state_info.csv',p=5,q=5):
    # Read the entire file into a single string
    file_path = str(pathlib.Path(os.getcwd()).parents[0] / 'cpp_sim' / filename)
    with open(file_path, 'r') as file:
        data = file.read()

    # Remove any possible empty lines
    lines = data.split('\n')
    lines = [line for line in lines if line]  # Remove empty strings

    # Calculate the number of lines
    num_lines = len(lines)
    print(f'num lines: {num_lines},P={p},Q={q}')
   

    # Convert the binary strings to a flat numpy array of integers
    binary_flat = np.frombuffer(''.join(lines).encode('utf-8'), dtype=np.uint8) - ord('0')

    # Reshape and transpose the array to get the desired shape (4, number_of_lines)
    print(f'p={p}q={q}')
    binary_array = binary_flat.reshape(num_lines, p+q).T

    return binary_array


#############################################################
##############  functions to deal with cpp sd info ##########
#############################################################

def convert_to_int(n):
    """ special conversion for cpp file """
    s=0
    for i,el in enumerate(n):
        s = s+ 2**i * int(el)
    return s

def get_sd_from_cpp_line(line,P=8,Q=4,last_2_info=True):
    """ line is a string from the CSV results file. alpha and newline char are clipped. """
    nums = line.split(',')[1:-1]
    if last_2_info:
        nums = nums[:-2]
    nums = [int(n) for n in nums][:2**(P+Q)]
    sd = np.empty((2**P,2**Q))
    for integer,density in enumerate(nums):
        if P + Q == 12:
            binary_num = '{0:012b}'.format(integer)[::-1]
        if P + Q == 11:
            binary_num = '{0:011b}'.format(integer)[::-1]
        if P + Q == 14:
            binary_num = '{0:014b}'.format(integer)[::-1]
        if P + Q == 13:
            binary_num = '{0:013b}'.format(integer)[::-1]
        if P + Q == 16:
            binary_num = '{0:016b}'.format(integer)[::-1]
        x=binary_num[:P]
        y=binary_num[P:]
        x=convert_to_int(x)
        y=convert_to_int(y)   # note these are indexes, not actual y / x values
        sd[x,y] = density
    sd = sd / np.sum(sd)
    return sd

################################################################################
############ coupling for cpp sim ##############################################
################################################################################
def get_cpp_coupling_info(F,P,Q,special = True):
    import sympy as sp
    # Finding the factors using sympy
    factors = sympy.divisors(F)
    print(f'factors of {F} are : {factors}')

    if special:
        intf = IntF(p=P, q=Q, F=F, special=9)
    else:
        intf = IntF(p=P, q=Q, F=F)

    # Get and print the coupling information
    coupling_information = intf.get_coupling()
    print("\n//Coupling Information:" + f" F = {F}, P = {P}, Q = {Q}")

    print("std::vector<float> get_biases(std::vector<bool> &state)\n{\n    " + f"std::vector<float> biases({P+Q}, 0.0);")
    for equation_num, coupling in enumerate(coupling_information):
        # Define symbolic variables
        X = sympy.IndexedBase('x')
        Y = sympy.IndexedBase('y')
        # Convert the lambda function to a sympy expression
        coupling_expr = sympy.sympify(coupling(X, Y))
        eq = str(coupling_expr)
        for idx in range(P):
            old = f'x[{idx+1}]'
            new = f'state[{idx}]'
            eq = eq.replace(old,new)
        for idx_y, idx_state in zip(range(1,Q+1),range(P,Q+P)):
            old = f'y[{idx_y}]'
            new = f'state[{idx_state}]'
            eq = eq.replace(old,new)
        print(f"    biases[{equation_num}] = {eq};")
    print("    return biases;\n}")

