import pickle
import numpy as np
import sys
sys.path.append("../tools/")
from clean_data import removeNaN
from sklearn.metrics import precision_score,recall_score
from remove_outliers import removeOutlier
from new_features import create_new_feature
from feature_selection import select_features
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from time import time

#The program that uses Decision Tree classification technique.

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r")
                        )
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
 'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person', 'from_messages',
 'from_this_person_to_poi', 'shared_receipt_with_poi']                

data_dict=removeNaN(data_dict)
finance=['salary']                      #change finance to get more outliers
data_dict=removeOutlier(data_dict,finance)
data_dict,new_feature=create_new_feature(data_dict)
features_list.append(new_feature)
k=12                                # Change K to get the k best features
new_features_list=list(select_features(data_dict,features_list,k))
new_features_list.insert(0, 'poi')


my_dataset=data_dict
data = featureFormat(data_dict, new_features_list)
labels, features = targetFeatureSplit(data)

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 20)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
a=clf.predict(features_test)
print np.intersect1d(labels_test,a)
print "The precision is",precision_score(labels_test,a)
print "The recall is",recall_score(labels_test,a)
print "The accuracy is",clf.score(features_test,labels_test)

dump_classifier_and_data(clf, my_dataset, new_features_list)
