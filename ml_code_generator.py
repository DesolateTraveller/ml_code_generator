#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------
import os
import sys
import io
import base64
import traceback
from PIL import Image
#----------------------------------------
from io import BytesIO
#----------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="ML Code Generator | v0.1",
                    layout="wide",
                    page_icon="💻",            
                    initial_sidebar_state="collapsed")
#----------------------------------------
st.title(f""":rainbow[ML Code Generator]""")
st.markdown(
    '''
    Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( :envelope: [Email](mailto:avijit.mba18@gmail.com) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''',
    unsafe_allow_html=True)
st.info('**An easy-to-use, open-source application to generate python codes for machine learning algorithms**', icon="ℹ️")
#----------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns((0.2,0.8))

with col1:

        data_source = st.sselectbox("Select the data source file extension:", [".csv file", ".xlsx file"])
        if data_source == ".csv file":
	        data_source = "csv"
        else:
	        data_source = "excel" 
