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

# %%
st.title("Visualization and predictive modeling of traffic stop data for the City of Charlotte")
st.markdown("Data sourced from [City of Charlotte Open Data Portal](https://data.charlottenc.gov/datasets/charlotte::officer-traffic-stops/explore)")
st.write("The data for this project was pulled 10/17/2022")


@st.cache
def open_clean_file():
    url = "https://drive.google.com/file/d/1P9wUxzlFcXs_sC0jBGBlcdyP56OMvK7W/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id=' +\
           url.split('/')[-2]
    stops = pd.read_csv(path)
    stops.drop(columns=["GlobalID", "OBJECTID"], inplace=True)
    stops["Month_of_Stop"] = pd.to_datetime(stops.Month_of_Stop)
    stops_ob = stops.select_dtypes(["object"])
    stops[stops_ob.columns] = stops_ob.apply(lambda x: x.str.strip())
    stops_filt = stops[~ stops.Officer_Race.isin(["2 or More",
                       "Not Specified"])].copy()
    stops_filt.drop(stops_filt[stops_filt.Driver_Race ==
                               "Other/Unknown"].index, inplace=True)
    stops_filt.drop(stops_filt[stops_filt.Reason_for_Stop ==
                               "Other"].index, inplace=True)
    stops_filt.dropna(inplace=True)
    stops_filt["Driver_Age"] = pd.cut(stops_filt.Driver_Age,
                                      bins=[0, 25, 35, 55, 200],
                                      labels=["Under 25",
                                              "25 to 34",
                                              "35 to 54",
                                              "55 and older"])
    stops_filt["Officer_Years_of_Service"] =\
        pd.cut(stops_filt.Officer_Years_of_Service,
               bins=[0, 5, 10, 15, 100],
               labels=["Less than 5",
                       "5 to 9",
                       "10 to 15",
                       "Over 15"])
    return stops_filt


stops_filt = open_clean_file()
st.write("View of the first few rows!")
with st.expander("Show first 5 rows of the dataframe"):
    st.table(stops_filt.head())

optionsb = [c for c in stops_filt.columns if c not in ['Month_of_Stop']]

targetgrp = st.selectbox(
    "Variable to build histogram from",
    optionsb)

st.write("You chose: ", targetgrp)

figa, axa = plt.subplots()
sns.histplot(data=stops_filt, y=targetgrp, ax=axa)
axa.set_title("Histogram of traffic stops by \"{0}\"\nin Charlotte, NC".
              format(targetgrp.replace("_", " ")))
axa.set_ylabel(targetgrp.replace("_", " "))

with st.expander("Show histogram: "):
    st.pyplot(figa)

choiceca = st.selectbox(
    "Which variable would you like to plot?",
    optionsb
)
choicecb = st.selectbox(
    "Which varible would you like to groupby?",
    optionsb
)

figc, axc = plt.subplots()
sns.countplot(data=stops_filt, y=choiceca, hue=choicecb, ax=axc)
axc.set_title("Histogram of {} grouped by {}".
              format(choiceca.replace("_", " "), choicecb.replace("_", " ")))
axc.set_ylabel(choiceca.replace("_", " "))
st.pyplot(figc)


# %%
@st.cache
def build_model():
    min_stops = stops_filt[["CMPD_Division",
                            "Driver_Race",
                            "Reason_for_Stop"]].copy()
    min_stops["Target"] = stops_filt["Result_of_Stop"]
    min_stops["Target"] = np.where(min_stops.Target == "Arrest", 1, 0)
    count_class_0, _ = min_stops.Target.value_counts()
    stops_arrest = min_stops[min_stops.Target == 1]
    stops_arrest_over = stops_arrest.sample(int(np.round(count_class_0 * .5)),
                                            replace=True)
    min_stops = pd.concat([min_stops, stops_arrest_over], axis=0)
    dt_X = min_stops.drop(columns="Target")
    dt_y = min_stops["Target"]
    ohedt = OneHotEncoder(handle_unknown="ignore")
    ohedt.fit(dt_X)
    dt_enc_X = ohedt.transform(dt_X).toarray()
    model = DecisionTreeClassifier(criterion="gini", max_depth=14,
                                   random_state=42)
    model.fit(dt_enc_X, dt_y)
    conditions = [list(min_stops[col].unique()) for col in min_stops]
    return model, ohedt, conditions


DT_model, encoder, inputs = build_model()


# %%
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
# %%
