import pickle
import sys
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# function that replaces NaN with 0

def removeNaN(data_dict):
    new=dict()

    for key in data_dict[data_dict.keys()[0]]:
        new[key]=0
    
    for key in data_dict.keys():
        for k in data_dict[key]:
            if data_dict[key][k]=='NaN':
                data_dict[key][k]=0
                new[k]=new[k]+1

    print "The number of NaN in the dataset are"
    for key in new.keys():
        print key,new[key]
    return data_dict
