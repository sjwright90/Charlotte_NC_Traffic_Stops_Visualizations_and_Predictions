#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:01:22 2022

@author: Samsonite
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from config.initialize import build_model, open_clean_file


# %%
stops_filt = open_clean_file()
DT_model, encoder, inputs = build_model()
st.markdown("**Predictive Modeling:**")
st.write("Choose the parameters of interest you would like to model")
divisionop = st.selectbox(
    "Which division was the driver stopped in?",
    inputs[0]
)
raceop = st.selectbox(
    "What race is the driver?",
    inputs[1]
)
reasonop = st.selectbox(
    "Why was the driver stopped?",
    inputs[2]
)
allops = [divisionop, raceop, reasonop]
columns = ["CMPD_Division", "Driver_Race", "Reason_for_Stop"]
to_df = dict(zip(columns, allops))
input_df = pd.DataFrame(to_df, np.arange(1))

input_enc = encoder.transform(input_df).toarray()
rawoutput = DT_model.predict(input_enc)
chances = DT_model.predict_proba(input_enc)

if rawoutput == 0:
    output = "**Unlikely to be arrested**"
else:
    output = "**Likely to be arrested**"
st.write(output)
st.write("Change of being arrested {0:.2f}  \nChance of not being arrested {1:.2f}".format(
    chances[0][1], chances[0][0]
))