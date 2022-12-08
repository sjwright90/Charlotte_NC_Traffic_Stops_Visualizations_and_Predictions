#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:01:22 2022

@author: Samsonite
"""
# %%
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from config.initialize import open_clean_file, read_markdown_file

# %%
st.title("Visualization and predictive modeling of traffic stop data for the City of Charlotte")
st.markdown("Data sourced from [City of Charlotte Open Data Portal](https://data.charlottenc.gov/datasets/charlotte::officer-traffic-stops/explore)")
st.write("The data for this project was pulled 10/17/2022")

with st.spinner("Getting your data ready"):
    stops_filt, stops_num = open_clean_file()
st.subheader("Understanding policing data in the Charlotte, NC metropolitan area")
intro_md = read_markdown_file("intro.md")
st.markdown(intro_md)
st.subheader("Getting familiar with the data:")
st.write("Here we are going to get familiar with the data set ",
         "we are working with. The data is a record of traffic ",
         "conducted in Charlotte, NC between January 2021 and ",
         "December 2021. Each row of the data frame is an individual ",
         "traffic stop and the columns provide information about that ",
         "stop. Take a look at the first few rows of the data frame ",
         "below to familiarize yourlself with the data")

with st.expander("Click to show first 5 rows of the dataframe"):
    st.table(stops_filt.head())

st.write("Great! As you can see there are a large number of data points ",
         "recorded for each stop! This is exciting since we will have ",
         "lots of data to work with!\n\nBelow are a few tools that ",
         "allow you to keep exploring the data set. Feel free ",
         "to poke around and see if anything sparks your interest. ",
         "When you are ready to move onto the next steps go to ",
         "one of the links on the left of the screen for graphing ",
         "tools and predictive modeling.")

st.subheader("Raw data")
with st.expander("Click here to look at some more raw data: "):
    st.write("Move the slider to select a random set of observations ",
    "from the data table.")
    size = st.select_slider("How many rows would you like to show?",
                            options=np.arange(1,11))
    guess = np.random.randint(stops_filt.shape[0]-1, size=size)
    st.table(stops_filt.iloc[guess, :])

st.subheader("Summary statistics")
with st.expander("Open to see summary statistics of the data: "):
    st.write("Summary statistics is a great place to start when getting ",
             "familiar with a data set. Below you can look at the ",
             "summary stats of both numeric and categorical columns in ",
             "the data")
    choicea = st.radio("Do you want to see categorical or numerical",
             options=["Categorical", "Numerical"])
    if choicea == "Categorical":
       summarystats = stops_num.describe(include="O").fillna("").astype("str")
    else:
        summarystats = stops_num.describe().fillna("").astype("str")
    
    st.table(summarystats)
# %%
