import pickle
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# Uses SelectKBest to get K best features
# Parameters are the dictionary, the features list and k
def select_features(data_dict,features_list,k):
                    
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
   

    train_data,test_data,train_labels,test_labels = train_test_split(
                 features, labels, test_size=0.3)

    selector = SelectKBest(k =k)
    selector.fit(train_data, train_labels)

    for i in range(len(selector.scores_)):
        print features_list[i+1],selector.scores_[i]

    selected_features = np.array(features_list[1:])[selector.get_support()]
    print selected_features
    return selected_features 
