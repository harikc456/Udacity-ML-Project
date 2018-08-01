import pickle
import sys
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# Uses IQR to find and remove outliers, finance is the list of features whose outliers is to be found
def removeOutlier(data_dict,finance):
    new=dict()

    for key in data_dict[data_dict.keys()[0]]:
        new[key]=0
    
    for key in data_dict.keys():
        for k in data_dict[key]:
            if data_dict[key][k]=='NaN':
                data_dict[key][k]=0
                new[k]=new[k]+1

    new_dict=dict()
    for key in data_dict[data_dict.keys()[0]]:
        new_dict[key]=list()
    
    for key in data_dict.keys():
        for k in data_dict[key]:
            new_dict[k].append(data_dict[key][k])

    q3_dict=dict()        
    for key in new_dict.keys():
        if key in finance:
            new_dict[key]=np.array(new_dict[key])
            q3_dict[key]=(np.percentile(new_dict[key], 75) - np.percentile(new_dict[key], 25))* 1.5
         
    li=set()
    for key in data_dict.keys():
        for k in data_dict[key]:
            try:
                if data_dict[key][k] >  q3_dict[k]:
                    li.add(key)
            except:
                continue
        
    li=list(li)        
    for i in range(len(li)):
        if data_dict[li[i]]['poi']==False:
            data_dict.pop(li[i],0)
            print li[i]


    features_list = ['poi','loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'director_fees',
                 'from_poi_to_this_person','from_this_person_to_poi']
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    print "The total number of datapoints are "+ str(len(data_dict))
    count=0
    for i in labels:
        if i!=0:
            count=count+1
    print "The total number of poi in the given dataset is "+ str(count)
    return data_dict
