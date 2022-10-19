#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:53:49 2022

@author: Samsonite
"""
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
os.chdir("/Users/Samsonite/Documents/DSBA UNCC/DSBA5122/DesignContest")
#%%
stops = pd.read_csv("Officer_Traffic_Stops.csv")
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

