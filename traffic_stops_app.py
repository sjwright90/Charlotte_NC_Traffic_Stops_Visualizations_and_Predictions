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
import seaborn as sns
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
    stops_filt["Driver_Age"] = pd.cut(stops_filt.Driver_Age,bins = [0,25,35,55,200],\
        labels = ["Under 25", "25 to 34", "35 to 54", "55 and older"])
    stops_filt["Officer_Years_of_Service"] = pd.cut(stops_filt.Officer_Years_of_Service,\
        bins = [0,5,10,15,100], labels = ["Less than 5", "5 to 9", "10 to 15", "Over 15"])
    return stops_filt

stops_filt = open_clean_file()
st.write("View of the first few rows!")
with st.expander("Show first 5 rows of the dataframe"):
    st.table(stops_filt.head())

optionsb = [c for c in stops_filt.columns if not c in ['Month_of_Stop']]

targetgrp = st.selectbox(
    "Variable to build histogram from",
    optionsb)

st.write("You chose: ", targetgrp)

figa, axa = plt.subplots()
sns.histplot(data = stops_filt, x = targetgrp, ax = axa)
axa.set_title("Histogram of traffic stops by \"{0}\"\nin Charlotte, NC".format(targetgrp.replace("_"," ")))
plt.xticks(rotation = 90)
st.pyplot(figa)

optionsa = [c for c in stops_filt.columns if not c in ['Month_of_Stop',\
    'Driver_Race']]
choice = st.selectbox(
    "Which variable would you like to plot against?",
    optionsa)

st.write("You chose: ", choice)

grouped = stops_filt.groupby(by = choice)["Driver_Race"].value_counts()
grouped = pd.DataFrame(grouped)
grouped.rename(columns={"Driver_Race":"Count"}, inplace = True)
grouped.reset_index(inplace = True)
fig, ax = plt.subplots()
sns.barplot(data = grouped, x = choice,\
    y = "Count", hue = "Driver_Race", ax = ax)
plt.xticks(rotation = 90)
ax.set_title("Histogram of stops by driver race grouped by \"{0}\"".format(choice.replace("_"," ")))

st.pyplot(fig)

#%%
