#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:01:22 2022

@author: Samsonite
"""
#%%
from functools import cache
from multiprocessing.resource_sharer import stop
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
@st.cache
def open_clean_file():
    url = "https://drive.google.com/file/d/1P9wUxzlFcXs_sC0jBGBlcdyP56OMvK7W/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    stops = pd.read_csv(path)
    stops.drop(columns = ["GlobalID", "OBJECTID"], inplace = True)
    stops["Month_of_Stop"] = pd.to_datetime(stops.Month_of_Stop)
    stops_ob = stops.select_dtypes(["object"])
    stops[stops_ob.columns] = stops_ob.apply(lambda x: x.str.strip())
    stops_filt = stops[~ stops.Officer_Race.isin(["2 or More", "Not Specified"])].copy()
    stops_filt.drop(stops_filt[stops_filt.Driver_Race == "Other/Unknown"].index, \
        inplace = True)
    stops_filt.drop(stops_filt[stops_filt.Reason_for_Stop == "Other"].index, inplace = True)
    stops_filt.dropna(inplace = True)
    return stops_filt

stops_filt = open_clean_file()
st.write("View of the first few rows!")
st.table(stops_filt.head())


choice = st.selectbox(
    "Which variable would you like to plot against?,
    "Officer_Race", "CMPD_Division", "Reason_for_Stop"))


grouped = stops_filt.groupby(by = choice)

fig, ax = plt.subplots()

ax.hist(grouped["Driver_Race"])

#%%
