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
stops = pd.read_csv("https://drive.google.com/file/d/1P9wUxzlFcXs_sC0jBGBlcdyP56OMvK7W/view?usp=sharing")
#%%
stops.drop(columns = ["GlobalID", "OBJECTID"], inplace = True)
#%%
stops_filt = stops[~ stops.Officer_Race.isin(["2 or More", "Not Specified"])].copy()

# alternative: stops.drop(stops[stops.Officer_Race.isin(["2 or More", "Not Specified"])].index)
#%%
stops_filt.dropna(inplace = True)
#%%
for col in stops_filt.columns:
    if not stops_filt[col].dtype == "int64":
        print(stops_filt[col].value_counts())
        print("\n")
 #%%   

stops_ob = stops_filt.select_dtypes(["object"])
stops_filt[stops_ob.columns] = stops_ob.apply(lambda x: x.str.strip())

#%%
stops_filt.drop(stops_filt[stops_filt.Reason_for_Stop == "Other"].index, inplace = True)
#%%

