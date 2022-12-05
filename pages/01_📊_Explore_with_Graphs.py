# %%

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config.initialize import open_clean_file

# %%
stops_filt = open_clean_file()

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