# %%

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config.initialize import open_clean_file

# %%
st.title("Graphical representations of policing data")
st.write("Visual analytics is an integral component of any ",
         "data analysis. It gives users a more comprehensible ",
         "view of the data than just seeing numbers and letters in ",
         "a table. Especially it allows us to start to tease patterns ",
         "out of the data. These initial patterns we can see ",
         "are a great way to get started on some more technical analysis, ",
         "it gives us a sense of where to target our future analysis. ",
         "Luckily, software programs, such as Python which this page is ",
         "built in, allow us to quickly and accurately plot large amounts of ",
         "data. Below are a few tools that will allow you to interact with ",
         "the Charlotte Traffic Stops data set in a graphical format. ",
         "Explore the data, see if you can find any interesting patterns ",
         "that might inspire you for further analysis.")
with st.spinner("Getting your data ready"):
    stops_filt, _ = open_clean_file()

st.subheader("Simple histogram")
st.write("Here you can make a simple histogram plot from any of the ", 
         "features of the data set. Don't forget to click the drop down ",
         "menu to show the plot!")
optionsb = [c for c in stops_filt.columns if c not in ['Month_of_Stop']]

targetgrp = st.selectbox(
    "Variable to build histogram from",
    optionsb)

st.write("You chose: ", targetgrp)

figa, axa = plt.subplots()
sns.histplot(data=stops_filt, y=targetgrp, ax=axa)
axa.set_title("Histogram of traffic stops by {0}\nin Charlotte, NC".
              format(targetgrp.replace("_", " ")))
axa.set_ylabel(targetgrp.replace("_", " "))

with st.expander("Click to show histogram you can close it when you are done: "):
    st.pyplot(figa)

st.subheader("Grouped histograms")
st.write("Here we will step it up a notch and look at histograms of ",
         "two variables from the data set. We accomplish this by ",
         "grouping the data and counting the number of observations in ",
         "each group. By plotting two variables against each other in this ",
         "way we get a better sense the relationships that exist ",
         "within our data.")

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