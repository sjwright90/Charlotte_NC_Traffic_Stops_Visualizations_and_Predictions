#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:53:49 2022

@author: Samsonite
"""
#%%
from multiprocessing.resource_sharer import stop
from sre_constants import SRE_INFO_PREFIX
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Load data from google drive (should be public, if not email me for link swrig109@uncc.edu)
#or download it from 
# https://data.charlottenc.gov/datasets/charlotte::officer-traffic-stops/explore

url = "https://drive.google.com/file/d/1P9wUxzlFcXs_sC0jBGBlcdyP56OMvK7W/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
stops = pd.read_csv(path)
#%%
# Remove unecessary columns
stops.drop(columns = ["GlobalID", "OBJECTID"], inplace = True)

#%%
# change month of stop to datetime object
stops["Month_of_Stop"] = pd.to_datetime(stops.Month_of_Stop)

#%%
# isolate object data types and string strip to remove trailing/leading spaces
stops_ob = stops.select_dtypes(["object"])
stops[stops_ob.columns] = stops_ob.apply(lambda x: x.str.strip())
#%%

# Remove stops with multiple officers or officer race not specified
stops_filt = stops[~ stops.Officer_Race.isin(["2 or More", "Not Specified"])].copy()
#remove stops where driver race is unknown
stops_filt.drop(stops_filt[stops_filt.Driver_Race == "Other/Unknown"].index, \
    inplace = True)
# remove stops where reason for stop is other
stops_filt.drop(stops_filt[stops_filt.Reason_for_Stop == "Other"].index, inplace = True)
#%%
# inspect number and location of NaN values
print(stops_filt.isna().sum())
# Drop stops with NaN
stops_filt.dropna(inplace = True)
#%%

# Look at distribution of categorical columns
for col in stops_filt.columns:
    if stops_filt[col].dtype == "O":
        print(stops_filt[col].value_counts())
        print("\n")
# better to look at graphical representation
#%%
# histograms of each categorical column, as we can see officers are most 
# often white and drivers are most often black
for col in stops_filt:
    if stops_filt[col].dtype == "O":
        stops_filt[col].hist()
        plt.xticks(rotation = 90)
        plt.title(col)
        plt.ylabel("Count")
        plt.grid(False)
        plt.show()
#%%
# same as above but with normalized bar chart

for col in stops_filt:
    if stops_filt[col].dtype == "O":
        temp = stops_filt[col].value_counts(normalize = True, sort = False)
        plt.bar(x = temp.index, height=temp)
        plt.xticks(rotation = 90)
        plt.title(col.replace("_", " "))
        plt.grid(False)
        plt.ylabel("Normalized Values")
        plt.show()

#%%
# lets see if officer race has any effect on race of the driver they stop

#group by officer race
leo_race = stops_filt.groupby(by = "Officer_Race")

for group in leo_race:
    plt.hist(group[1].Driver_Race.sort_values())
    plt.title("Officer Race: " + group[0])
    plt.ylabel("Count")
    plt.xlabel("Driver Race")
    plt.show()
# as we can see black drivers are stopped most often across all officer races, excepting 
# Asian/Pacific Islander where white drivers are the most stopped
#%%
# now let's see how CMPD division effects the race of the driver being stopped

cmpd_loc = stops_filt.groupby(by = "CMPD_Division")

for grp in cmpd_loc:
    plt.hist(grp[1].Driver_Race.sort_values())
    plt.title("CMPD Division: " + grp[0])
    plt.ylabel("Count")
    plt.xlabel("Driver Race")
    plt.show()

#CMPD divisions Providence and South are the only two divisions where the driver being stoped
#is not most likely to be black, this likely is a reflection of the population in those two
#locations in Charlotte
#%%
# finally let us see what the relationship between result of a stop and driver race is

outcome = stops_filt.groupby(by = "Driver_Race")
for grp in outcome:
    plt.hist(grp[1].Result_of_Stop.sort_values())
    plt.title("Driver Race: " + grp[0])
    plt.ylabel("Count")
    plt.xlabel("Result of Stop")
    plt.show()
# interesting results, it might be more informative to see the normalized number of arrests
# per driver race
#%%
arrests_byrace = stops_filt.query("Result_of_Stop == 'Arrest'")["Driver_Race"].value_counts()\
    /stops_filt.Driver_Race.value_counts() * 100

fig, ax = plt.subplots()
ax.bar(x = arrests_byrace.index, height=arrests_byrace)
ax.set_ylabel("Percent of Total")
ax.set_xlabel("Driver Race")
ax.set_title("Percent of stops that result\nin arrest by race")
plt.show()

#This gives a better picture, while accross all races the percent of arrests is
#small (<4%) it is much higher for black drivers, who are more than twice as 
#likely to be arrested as any other race
#%%
# time to build some simple predictive models
# we will keep it basic here, just a DecisionTree, a Logistic Regression model,
# and a Naive Bayes classifier
# warning: naive Bayes is a good classifier, but reported to be a bad predictor
from sklearn.tree import DecisionTreeClassifier #DT classifier, 
from sklearn.model_selection import train_test_split #to divide the data
from sklearn.linear_model import LogisticRegression #logreg model
from sklearn.preprocessing import OneHotEncoder #label encoding for naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
#%%

#for Naive bayes we really only want categorical variables, so let's drop the numeric
#columns from the dataset
nb_stops = stops_filt.copy()
todrop = [col for col in nb_stops if nb_stops[col].dtype != "O"]
nb_stops.drop(columns = todrop, inplace = True)
# and break off predictors and target
nb_vars = nb_stops.drop(columns = "Result_of_Stop")
nb_trg = nb_stops["Result_of_Stop"].array
nb_trg = nb_trg.reshape(-1,1)
#%%
# then we want to divide into training and testing sets

nb_X_train, nb_X_test, nb_y_train, nb_y_test = train_test_split(nb_vars, nb_trg, test_size=0.2, \
    random_state = 32, 
    stratify=nb_trg)

#then use one hot encoder to get dummy variables
#one encoder for the predictors and one for the target
ohe_nb_X = OneHotEncoder(handle_unknown="ignore")
ohe_nb_y = OneHotEncoder(handle_unknown="ignore")

#fit predictor training, then transform on both predictor train and test
ohe_nb_X.fit(nb_X_train)
nb_X_train_enc = ohe_nb_X.transform(nb_X_train).toarray()
nb_X_test_enc = ohe_nb_X.transform(nb_X_test).toarray()

#ditto for target
ohe_nb_y.fit(nb_y_train)
nb_y_train_enc = ohe_nb_y.transform(nb_y_train).toarray()
nb_y_test_enc = ohe_nb_y.transform(nb_y_test).toarray()

#%%
#then we fit the naive bayes classifier, which as it turns out does not
#need the target variables to be encoded
nb_y_train_r = np.ravel(nb_y_train)
nb_mclass = MultinomialNB()

nb_mclass.fit(nb_X_train_enc, nb_y_train_r)
#%%
#and look at the result
nb_y_test_r = np.ravel(nb_y_test)
print(nb_mclass.score(nb_X_test_enc, nb_y_test_r))

#0.65, pretty bad, let's dig a little deeper
#%%
nb_y_pred = nb_mclass.predict(nb_X_test_enc)
nb_acc = accuracy_score(nb_y_test_r, nb_y_pred)
ConfusionMatrixDisplay.from_estimator(nb_mclass, nb_X_test_enc, nb_y_test_r)
plt.xticks(rotation = 90)
plt.show()
'''from the confusion matrix we can see that the classifier does pretty well
for Verbal Warnings and Citations, which unsurprisingly are the two features
with the highest occurence. There are options from here, one could balance the data
set to upsample arrest, no action taken, and written warning. Alternatively one 
could build a Binomial model with arrest vs not arrest, would probaly still want to
upsample the arrests in this case. One could also reduce the number of predictor
variables. For the moment we will leave this as is and move on to other classifiers'''
#%%
'''Next we will explore the simple logistic regression classifier,
in this instance we can leave all predictor variables in, but
we will drop the months column again since we are not doing a time
series analysis here'''
#we can bin the two continuous variables Officer_Years_of_Service and Driver_Age
stops_filt["Officer_Years_of_Service"] = pd.cut(stops_filt.Officer_Years_of_Service,\
     bins = [0,5,10,15,100],labels = ["<5years", "5-10years", "10-15years", "15+years"])
#%%
stops_filt["Driver_Age"] = pd.cut(stops_filt.Driver_Age, bins = [0,25,35,55,200], \
    labels = ["under25", "25-34","35-54", "55+"])
#%%
lg_X = stops_filt.drop(columns = ["Month_of_Stop", "Result_of_Stop"])
lg_y = stops_filt["Result_of_Stop"]

lg_X_train, lg_X_test, lg_y_train, lg_y_test = train_test_split(lg_X, lg_y, test_size=0.3, \
    random_state=42, stratify=lg_y)

ohe_lg_X = OneHotEncoder(handle_unknown="ignore")
ohe_lg_X.fit(lg_X_train)
lg_enc_X_train = ohe_lg_X.transform(lg_X_train).toarray()
lg_enc_X_test = ohe_lg_X.transform(lg_X_test).toarray()

#%%
lg_stops = LogisticRegression(solver="saga", n_jobs=-1, max_iter=200, warm_start=True)

lg_stops.fit(lg_enc_X_train, lg_y_train)

#%%
score = lg_stops.score(lg_enc_X_test, lg_y_test)
print(score)
'''at 0.68 we are slightly better than Naive Bayes, but hardly, for comparison our baseline is 
0.53 which is the normalized number of Verbal Warnings, so, better than baseline but not much.'''
ConfusionMatrixDisplay.from_estimator(lg_stops, lg_enc_X_test, lg_y_test)
plt.xticks(rotation = 90)
plt.show()

'''And looking at the confusion matrix we see a similar issue, with Citation Issued and Verbal Warning
being overpredicted, likely due to their greater abundance. Unfortunately the sklearn logistic
regression model does not allow us to exame the relative importance of the coefficients, and it is 
not very straightforward to do with statsmodels. I ran multinomial regression in R (see r script in
 this repo) and little more was gleaned.
 As with earlier, there are more steps to take here, filtering variables, testing different parameters, 
 and so on, however we will keep moving on to our decision tree'''
#%%
'''Decision tree classifier'''
#instantiate a decision tree
stops_dt = DecisionTreeClassifier(max_depth=10)

#fit on the train test set created above, potential for data leakage since the test sets
#have already been used, but in this instance that is ok

stops_dt = stops_dt.fit(lg_enc_X_train, lg_y_train)

#%%
score = stops_dt.score(lg_enc_X_test, lg_y_test)
print(score)
#out of the box decision tree, with 10 levels, returns a 70% accuracy,
#better so far than previous models, let's tune it a little bit
#%%
'''use grid search CV to test hyperparaments'''
from sklearn.model_selection import GridSearchCV
#%%
dec_tree = DecisionTreeClassifier(random_state=42)
criterion = ["gini","entropy"]
max_depth = [14,18,22]
parameters = dict(criterion = criterion, max_depth = max_depth)


clf_GS = GridSearchCV(stops_dt_a, parameters, n_jobs=-1, cv = 8)
clf_GS.fit(lg_enc_X_train,lg_y_train)

#%%
print("Best criterion: ", clf_GS.best_estimator_.get_params()["criterion"])
print("Optimal max_depth: ", clf_GS.best_estimator_.get_params()["max_depth"])
print("Best score: ", clf_GS.best_score_)
# still only 70% so no real chaneg from an out of the box model
#%%
#lets see perfomance on the training set
score = clf_GS.score(lg_enc_X_test, lg_y_test)
print(score)
ConfusionMatrixDisplay.from_estimator(clf_GS, lg_enc_X_test, lg_y_test)
plt.xticks(rotation = 90)
plt.show()
#and no suprises again, Verbal Warning and Citation Issued 
# are being over predicted, however, it is good to see that
#the test set has similar accuracy to the training set, 
#this suggests that over fitting is not an issue
#%%
'''Quick plot of relative feature importances, interestingly it appears
as though officer years of service, reason for stop, and CMPD division
are the most important in prediciting the outcome of a stop. The model is not 
incredibly accurate, still nearly twice as accurate as a guess, but one
could argue that race does not actually have much impact on the outcome of 
a traffic stop'''
fig, ax = plt.subplots(figsize = (50,50))
ax.barh(ohe_lg_X.get_feature_names_out(), stops_dt.feature_importances_)
ax.tick_params(axis = "y", which = "major", labelsize = 25)
plt.show()
#%%
'''Text representation of teh graph above'''
for feat, importance in zip(ohe_lg_X.get_feature_names_out(), stops_dt.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i = importance))
#%%
import xgboost as xgb
from sklearn.metrics import auc, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from scipy.stats import uniform, randint
#%%
def report_beat_scores(results, n_top = 3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("mean validation score: {0:.3f}(std: {1:.3f})".format(\
                results["mean_test_score"][candidate],
                results["std_test_score"][candidate]))
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")
#%%
#simple xgboost model
xgb_m1 = xgb.XGBClassifier(objective = "multi:softprob", \
    random_state = 42, use_label_encoder = False)
params = {
    "colsample_bytree":uniform(0.7, 0.3),
    "gamma":uniform(0,0.5),
    "learning_rate":uniform(0.03, 0.3),#default 0.1
    "max_depth":randint(14,18), #default 3
    "n_estimators":randint(100,150), #default 100
    "subsample":uniform(0.6,0.4)
}

search =RandomizedSearchCV(xgb_m1, param_distributions=params, random_state=42,\
    n_iter=200, cv = 8, verbose=1, n_jobs=-1, return_train_score=True)
search.fit(lg_enc_X_train, lg_y_train)
report_beat_scores(search.cv_results_)
#%%
#%%
xgb_m1.fit(lg_enc_X_train, lg_y_train)
score = xgb_m1.score(lg_enc_X_test, lg_y_test)
print(score)
#%%
#