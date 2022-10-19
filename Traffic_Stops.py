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
from sklearn.preprocessing import LabelEncoder #label encoding for naive bayes
#from sklearn.naive_bayes import 
#%%