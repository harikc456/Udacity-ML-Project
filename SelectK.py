import pickle
import sys
sys.path.append("../tools/")
from clean_data import removeNaN
from remove_outliers import removeOutlier
from new_features import create_new_feature
from feature_selection import select_features
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import matplotlib.pyplot as plt

# Program that checks the score for all the k values and plot it

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r")
                        )
features_list = ['poi','salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options'
                 , 'other', 'long_term_incentive',
 'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']                

data_dict=removeNaN(data_dict)
finance=['salary']                      #change finance to get more outliers
data_dict=removeOutlier(data_dict,finance)
data_dict,new_feature=create_new_feature(data_dict)
features_list.append(new_feature)
k=1
index=[]
k_list=[]
i=0
while k < 20:
    index.append(k)
    new_features_list=list(select_features(data_dict,features_list,k))
    new_features_list.insert(0, 'poi')

    my_dataset=data_dict
    data = featureFormat(data_dict, new_features_list)
    labels, features = targetFeatureSplit(data)

# Craeting, training and validationg GaussianNB classifier

    from sklearn.naive_bayes import GaussianNB 
    clf = GaussianNB()

    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

    clf.fit(features_train,labels_train)
    k_list.append(clf.score(features_test,labels_test))
    k=k+1
    
index=np.array(index)
k_list=np.array(k_list)
fig = plt.figure()
plt.bar(index,k_list)
plt.xlabel("Values Of K")
plt.ylabel("Scores for K value")
#plt.show()
plt.savefig('Plot-2.png')
