# %%

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# %%
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