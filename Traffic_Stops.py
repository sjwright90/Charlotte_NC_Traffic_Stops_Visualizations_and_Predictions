#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:53:49 2022

@author: Samsonite
"""
#%%
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt

#%%
# Load data from google drive (should be public, if not email me for link swrig109@uncc.edu)
url = "https://drive.google.com/file/d/1P9wUxzlFcXs_sC0jBGBlcdyP56OMvK7W/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
stops = pd.read_csv(path)
#%%
# Remove unecessary columns
stops.drop(columns = ["GlobalID", "OBJECTID"], inplace = True)
#%%
# Remove stops with multiple officers or officer race not specified
stops_filt = stops[~ stops.Officer_Race.isin(["2 or More", "Not Specified"])].copy()

# alternative: stops.drop(stops[stops.Officer_Race.isin(["2 or More", "Not Specified"])].index)
#%%
# inspect number and location of NaN values
stops_filt.isna.sum()
# Drop stops with NaN
stops_filt.dropna(inplace = True)
#%%
# Look at distribution of categorical columns
for col in stops_filt.columns:
    if not stops_filt[col].dtype == "int64":
        print(stops_filt[col].value_counts())
        print("\n")
 #%%   
# isolate object data types and string strip to remove trailing/leading spaces
stops_ob = stops_filt.select_dtypes(["object"])
stops_filt[stops_ob.columns] = stops_ob.apply(lambda x: x.str.strip())

#%%
# drop Other category from Reason for Stop

stops_filt.drop(stops_filt[stops_filt.Reason_for_Stop == "Other"].index, inplace = True)
#%%
# change month of stop to datetime (might want to do this earlier)
stops_filt["Month_of_Stop"] = pd.to_datetime(stops_filt.Month_of_Stop)
#%%
# histograms of each categorical column, as we can see officers are most often white and drivers are
# most often black
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
# alterantive: stops_filt["Driver_Race"].hist(by = stops_filt["Officer_Race"])


leo_race = stops_filt.groupby(by = "Officer_Race")

for group in leo_race:
    plt.hist(group[1].Driver_Race)
    plt.title(group[0])
    plt.ylabel("Count")
    plt.xlabel("Driver Race")
    plt.show()
\

#%%