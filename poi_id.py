#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np
import poi_email_addresses as p_email
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'deferral_payments', 
                 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees','to_messages', 
                 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi'
                 ] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
    

print
print 'Number of Rows:',len(data_dict.keys())
print "Number of Features: ", len(data_dict["LAY KENNETH L"].keys())
print


    
### Task 2: Remove outliers

print "TOTAL:", data_dict["TOTAL"]["salary"]
print


### Fisrt outlier is the TOTAL that is on the dict just need to pop it
data_dict.pop("TOTAL")

### Second outlier is the THE TRAVEL AGENCY IN THE PARK just need to pop it
print data_dict['THE TRAVEL AGENCY IN THE PARK']['salary']
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

data = featureFormat(data_dict, features_list, remove_all_zeroes=False)
poi = np.array(data[:,0])

print
print "Percent of pois: {:.2f}%".format(poi.mean()*100)
print

print 
print "Removing lot of keys without a lot of values"
for person in data_dict.keys():
        person_d = data_dict[person]
        num_of_nans = 0
        for key in person_d:
            if person_d[key]=='NaN':
                num_of_nans += 1
        if num_of_nans > 15:
            print person
            data_dict.pop(person)
print
print 'Number of Rows:',len(data_dict.keys())
print "Number of Features: ", len(data_dict["LAY KENNETH L"].keys())
print

print "Features: ", features_list

data = featureFormat(data_dict, features_list, remove_all_zeroes=False)

total_payments = np.array(data[:,3])
total_stock_value = np.array(data[:,8])
long_term_incentive = np.array(data[:,12])
salary = np.array(data[:,1])
restricted_stock = np.array(data[:,13])
expenses = np.array(data[:,9])
bonus = np.array(data[:,5])


def num_of_nan(values):
    return (values==0).sum()

for pos, feature in enumerate(features_list):
    if pos==0:
        continue
    feature_datas = np.array(data[:,pos])
    feature_nas = num_of_nan(feature_datas)
    if feature_nas>100:
        features_list.remove(feature)
    print "{} NaNs: {}".format(feature, feature_nas)


data = featureFormat(data_dict, features_list, remove_all_zeroes=False)


#



plt.scatter(salary, total_stock_value)
plt.xlabel("salary")
plt.ylabel("total_stock_value")
plt.show();

plt.scatter(salary, total_payments)
plt.xlabel("salary")
plt.ylabel("total_payments")
plt.show();

plt.scatter(salary, expenses)
plt.xlabel("salary")
plt.ylabel("expenses")
plt.show();

plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show();

print "LAY KENNETH L salary:", data_dict["LAY KENNETH L"]["salary"]
print "LAY KENNETH L total_payments:", data_dict["LAY KENNETH L"]["total_payments"]
print


### The others outliers are needed for this project

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# check if the e-mail is from a poi person
total = total_stock_value+total_payments
long_term_stock=long_term_incentive+restricted_stock
salary_bonus = salary+bonus
total_spent = total_payments+expenses
for pos, person in enumerate(data_dict.keys()):
    data_dict[person]['long_term_stock']=long_term_stock[pos]
    data_dict[person]['total_spent']=total_spent[pos]
    
my_dataset = data_dict
features_list.append('total_spent')
features_list.remove('salary')
features_list.append('long_term_stock')
features_list.remove('long_term_incentive')
features_list.remove('restricted_stock')

### Extract features and labels from dataset for local testing

data = featureFormat(data_dict, features_list)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
clfs = [rfc, abc]

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.model_selection import GridSearchCV  

#kbest = SelectKBest(f_classif, k=1).fit(features, labels)
kbest = SelectKBest(f_classif, k=2).fit(features, labels)
#kbest = SelectKBest(f_classif, k=3).fit(features, labels)




for pos, score in enumerate(kbest.scores_):
    print "{}: {}".format(features_list[pos+1],score)

features = kbest.transform(features)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=5, random_state=42).split(features, labels)

features_train = []
features_test  = []
labels_train   = []
labels_test    = [] 

for train_idx, test_idx in split: 
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )   


#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)


params = [
            {'bootstrap': (False, True),
             'criterion': ('gini','entropy'), 
             'max_features': ['auto','sqrt','log2'], 
              'class_weight': ['balanced', 'balanced_subsample', None], 
              'n_estimators': [50,100], 
              'max_depth':[2, 3], 
              'min_samples_split':[50,100]},
             
             {'algorithm':('SAMME','SAMME.R'), 
              'random_state':[42,50,100,150],
              'n_estimators':[20,50,100,150,200], 
              'learning_rate':[.20,.30,.40,.50,.60,.70,.80,.90,1.0]}
         ]

def best_estimator(clf, param):
    clf = GridSearchCV(clf, param, cv=5).fit(features_train, labels_train)
    return clf.best_estimator_



#clf = best_estimator(clfs[0], params[0])
clf = best_estimator(clfs[1], params[1])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)