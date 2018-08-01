import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )
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
       
