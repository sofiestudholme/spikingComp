""" 


ACF and IEI helper funcs 


"""

from boltzmann_helper import *


def autocorr2(lags,data):
    acorr = np.empty(len(lags))
    mean = np.sum(data) / len(data) 
    print(mean)
    var = np.sum((data - mean)**2) / len(data)
    data = data - mean
    for n, l in enumerate(lags):
        l = round(l)
        c = 1 # Self correlation
        if (l > 0):
            tmp = data[:len(data)-l]*data[l:]
            # tmp = data * np.roll(data,l)
            c = np.mean(tmp) / var 
            
        acorr[n] = c
    return acorr

def cross_correlation(s1, s2, lags):

    assert len(s1) == len(s2), "s1 and s2 must be the same length"
    mean_s1, mean_s2 = np.mean(s1), np.mean(s2)
    s1_centered, s2_centered = s1 - mean_s1, s2 - mean_s2
    
    correlations = []
    for lag in lags:
        corr = np.dot(s1_centered[:-lag], s2_centered[lag:]) / len(s1_centered[:-lag])
        correlations.append(corr)
    
    return np.array(correlations)


def convert_to_inter_event_intervals(E):
    # Find the indices of 1s
    event_indices = np.where(E == 1)[0]
    
    # Check if there are no events or only one event
    if len(event_indices) < 2:
        return []
    
    # Calculate inter-event intervals
    inter_event_intervals = np.diff(event_indices)
    
    return inter_event_intervals
