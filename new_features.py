#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# creates a new feature to_from_poi_ratio
# Ratio is calculated by dividing from_this_person_to_poi feature by from_poi_to_this_person
def create_new_feature(data_dict):

    new=dict()

    for key in data_dict[data_dict.keys()[0]]:
        new[key]=0
    
    for key in data_dict.keys():
        for k in data_dict[key]:
            if data_dict[key][k]=='NaN':
                data_dict[key][k]=0
                new[k]=new[k]+1

    for key in data_dict.keys():
        try:
            data_dict[key]['to_from_poi_ratio']=float(data_dict[key]['from_this_person_to_poi'])/float(data_dict[key]['from_poi_to_this_person'])
        except:
            data_dict[key]['to_from_poi_ratio']=0.0

    return data_dict,'to_from_poi_ratio'
