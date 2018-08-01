
## Introduction

The dataset which is used for this project is the Enron dataset. Enron scandal is one of the world's biggest bankruptcy cases. A lot of employees of this corporation where found to be guilty and arrested. The objective of this project is to build a classifier that can classify whether a person is POI (Person Of Interest, i.e whether the person is guilty or not). This is performed by training a classifier using the given data points which contains numerous features about a given person. The featues can broadly classified into two types :-
 * Financial features
 * Email features
 
Financial features contains 14 features all of which are in US dollars. <br>
Email features contains 6 features all of which are numbers except the email address which is a string.<br>
Together we have 20 features and labels for classifying the person. The labels are either 0 or 1 , which signifies whether a person is a poi or not.

## Exploring The Dataset

By running  the no_of.py program attached with this file , we get the following information :-
 * There are 146 data points 
 * There are 18 person of interest.
 
This would mean that there are 128 employees who are not of interest. Hence we can conclude that this dataset is heavily skewed towards not_poi class label. 

### Missing Values 

Next by running the clean_data.py, i replaced all the NaN values with 0. Also the number of NaN for each feature is also calculated and print out. The output of the program is shown below.

<pre>
The number of NaN in the dataset are
salary 51
to_messages 60
deferral_payments 107
total_payments 21
loan_advances 142
bonus 64
email_address 35
restricted_stock_deferred 128
total_stock_value 20
shared_receipt_with_poi 60
long_term_incentive 80
exercised_stock_options 44
from_messages 60
other 53
from_poi_to_this_person 60
from_this_person_to_poi 60
poi 0
deferred_income 97
expenses 51
restricted_stock 36
director_fees 129
</pre>

We can see that every feature except the labels is riddled with missing values.

## Outliers In The Finance Features

The technique used here, for finding outlier is the interquartile range (IQR). Any value above 1.5 times the IQR is considered as an outlier. By running remove-outliers.py using all the 14 finance features we get 115 outliers. Since we have only 146 datapoints removing all the outliers will result in insufficent data for training and testing a model. Hence only outliers in salary are removed. There are 11 outliers in salary of which 5 of them are POI. Removing 5 poi means that the our data will be even more skewed, therefore all 5 outlier in salary feature is kept in the dictionary. These people were classified and removes as outliers

    WHALLEY LAWRENCE G
    SHERRIFF JOHN R
    DERRICK JR. JAMES V
    FREVERT MARK A
    PICKERING MARK R
    TOTAL

## Creating New Features

A new feature to_from_poi_ratio is created by dividing the feature from_this_person_to_poi by the feature from_poi_to_this_person. This process is done in the new_features.py. This program then returns both the new data dictionary and the new feature name.

## Feature Selection

A dataset can be riddlied with unwanted data that may diminish the performance of a classifier. Sklearn provides numerous ways to do this. In this project SelectKBest is chosen, as the name signifies it chooses K best features out of all the features given as input . The program feature_selection prints the following result before returning k best fetaures

<pre>
salary 20.8498024705
deferral_payments 1.04216019139
total_payments 7.30464936171
loan_advances 6.85567010309
bonus 8.1256478952
restricted_stock_deferred 0.101782199442
deferred_income 11.7500546832
total_stock_value 21.7705392079
expenses 4.859873928
exercised_stock_options 19.5325659241
other 10.187646113
long_term_incentive 16.6163053274
restricted_stock 14.5124834016
director_fees 1.81038175609
to_messages 0.217882980087
from_poi_to_this_person 1.67459797658
from_messages 0.137419269468
from_this_person_to_poi 1.51725891825
shared_receipt_with_poi 3.09897210321
to_from_poi_ratio 3.92146923953
</pre>

The result shows the feature along with score given by the SelectKBest(). Higher score tend to contribute more to the accuracy of the classifier.

The new feature to_from_poi_ratio has a importance of 3.92146923953, but it is not important as the features given below i.e the eleven best features out of the feature list. So it is safe to say that the new feature to_from_poi_ratio has a minimal influence on precision and recall.

<img src="Plot-2.png"></img>


Above given is a plot K values and their corresponding score, the value k is chosen as 11 because k=11 has one of the highest scores in the plot. Eleven features selected by SelectKBest() are :-

['salary' 'total_payments' 'loan_advances' 'bonus' 'deferred_income'
 'total_stock_value' 'expenses' 'exercised_stock_options' 'other'
 'long_term_incentive' 'restricted_stock']

## Comparing Various Classifying Algorithm

Algorithms compared in this section are :-
 * Gaussian Naive Bayes
 * Bernoulli Naive Bayes
 * Decision Tree
 * Support Vector Machine

### Gaussian Naive Bayes
 * Training time - 0.0 s
 * Precision -  0.66
 * Recall - 0.4
 * Accuracy - 90.4%
 
### Bernoulli Naive Bayes
 * Training time - 0.108 s
 * Precision - 0.176
 * Recall - 0.6
 * Accuracy - 61.9%
 
### Decision Tree
 * Training time - 0.0 s
 * Precision - 0.142
 * Recall - 0.2
 * Accuracy - 76.19%
 
### Support Vector Machine
 * Training time - 0.047 s
 * Precision - 0.0 
 * Recall - 0.0
 * Accuracy - 88.09%
 
<b>All the above algorithms use their default parameters </b>

## Tuning Different Parameters

Tuning is a process were various parameters of the chosen alogorithm is varied to enhance the accuracy of the model. Not properly tuning a classifier can cause overfitting or underfitting which can result in a lot of unwanted errors in the classification model. Below given are few scenario how tuning can change the performance of a model. 

### Decision Tree #1

 * min_samples_split = 2
 * Precision - 0.125
 * Recall - 0.2
 * Training time - 0.016 s
 * Accuracy - 73.80%
 
### Decision Tree #2

 * min_samples_split = 20
 * Precision - 0.1
 * Recall - 0.2
 * Training time - 0.003 s
 * Accuracy - 69.04%
 
### Decision Tree #3

 * min_samples_split = 50
 * Precision - 0.33
 * Recall - 0.2
 * Training time - 0.005 s
 * Accuracy - 85.71%
 
### SVC #1 

 * kernel = rbf
 * C = 100000
 * Precision - 0.0 
 * Recall - 0.0
 * Training time - 0.005 s
 * Accuracy - 85.71%
 
### SVC #2

 * kernel = poly
 * C = 100000
 * Precision - 0.119
 * Recall - 1.0
 * Training time - 0.031 s
 * Accuracy - 11.90%
 
### SVC #3

 * kernel = sigmoid
 * C = 100000
 * Precision - 0.0 
 * Recall - 0.0
 * Training time - 0.033 s
 * Accuracy - 80.95%
 
### SVC #4 

 * kernel = rbf
 * C = 10
 * Precision - 0.0
 * Recall - 0.0
 * Training time - 0.06 s
 * Accuracy - 88.09%  

The scenarios where precision and recall are zero is due to the following error-<br> UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.

From testing various classifying algorithm with varying parameters, we can see that the algorithm with highest score is the Gaussian Naive Bayes. Therefore this technique will be implemented in the poi_id.py.

## Feature Scaling

Feature scaling is technique used to normalize the value of a given dataset between 0 or 1 (or between any two smaller values than it is). It helps when these feature are compared with other features which does not match in scale. No feaure scaling was used in this project. 

## Evaluation Metrics

Evaluation metrics such as precision and recall is deployed to evaluate the classification model.

Precision is number of correctly classified data out of all the positive predictions made by the classifier. So precision gives us a value between 0.0 and 1.0 which shows how many prediction were correct out of all the positive prediction.  
 
Precision = True Positive/(True Positive + False Positive)
       
Recall is the fraction of data that is classified as postive out of all the postive data points. Recall gives the fraction that represents how much of the positive data were classified correctly. 
       
Recall = True Positive/(True Positive + False Negative)

Both Precison and Recall helps us to evaluate how good a model is.

Validation is essential step for getting an estimate of how well does our classifier fare on a independent dataset. Also it also provides us a means for checking for overfitting. Validation performed on the attached program is the cross validation, the test data is the 30% of the while data, while rest of 70% is used to train the model.

## Program poi_id.py

poi_id.py uses Gaussian Naive Bayes as the classification technique, it runs successfully and produces the files my_classifier.pkl, my_dataset.pkl, and my_feature_list.pkl which contains the classification model, the dataset used and features used respectively. After dumping those data in above mentioned files, tester.py is run and results are shown below:-
<pre>
GaussianNB(priors=None)
	Accuracy: 0.84907	Precision: 0.46310	Recall: 0.35450	F1: 0.40159	F2: 0.37194
	Total predictions: 14000	True positives:  709	False positives:  822	False negatives: 1291	True negatives: 11178
</pre>    

We can see that Precision is 0.51484 and Recall is 0.35550, which satisfies the condition presented in the Rubric.

## Index Of Files
<pre>
Bernoulli.py  - The program that uses Bernoulli Naive Bayes classification technique.
clean_data.py - The program that checks for missing values and fills them out.
feature_selection.py - The program that selects K best features depending upon the value of k.
Gaussian.py - The program that uses Gaussian Naive Bayes classification technique.
new_features.py - The program that creates a new feature and add them to the feature list.
no_of.py - Counts the number of data points and poi's.
remove_outliers - The program which finds the outliers and removes them.
SelectK.py - Program that checks the score for all the k values and plot it.
SVM.py - The program that uses Support Vector Machine classification technique.
Tree.py - The program that uses Decision Tree classification technique.
</pre>
