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
st.title("Predictive modeling")
st.write("Predictive models allow us to infer what will ",
         "happen in a future scenario given what has happened ",
         "in the past. While not infallible, these models can ",
         "give us insight and aid in decision making processes. "
         "On this page you can explore a decion tree model which ",
         "was built from the data you have been using up to this point. ",
         "The model will predict the likelyhood of an arrest happening ",
         "given certain parameters of a traffic stop. Explore the model and ",
         "see whay you make of the results.")
with st.spinner("Setting up predictive model"):
    stops_filt, _ = open_clean_file()
    DT_model, encoder, inputs = build_model(stops_filt)

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
figa, axa = plt.subplots()
axa.barh(y=["Probability of no arrest","Probability of arrest"],
         width=[chances[0][0],chances[0][1]])
axa.set_title("Probablity of being arested during this traffic stop")
st.pyplot(figa)