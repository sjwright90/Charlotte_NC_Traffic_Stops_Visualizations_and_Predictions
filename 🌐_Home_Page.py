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
from config.initialize import open_clean_file

# %%
st.title("Visualization and predictive modeling of traffic stop data for the City of Charlotte")
st.markdown("Data sourced from [City of Charlotte Open Data Portal](https://data.charlottenc.gov/datasets/charlotte::officer-traffic-stops/explore)")
st.write("The data for this project was pulled 10/17/2022")

with st.spinner("Getting your data ready"):
    stops_filt = open_clean_file()
st.subheader("Understanding policing data in the Charlotte, NC metropolitan area")
st.markdown('''

''')
st.write("View of the first few rows!")
with st.expander("Show first 5 rows of the dataframe"):
    st.table(stops_filt.head())
