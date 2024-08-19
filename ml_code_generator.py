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

        data_source = st.selectbox("Select the data source file extension:", [".csv file", ".xlsx file"])
        if data_source == ".csv file":
	        data_source = "csv"
        else:
	        data_source = "excel" 
		
        #--------------------------------------------------------------------
		
        path = st.text_input("Enter the input data file path here:", "Desktop/")
		
		#--------------------------------------------------------------------
        
        algorithm = st.selectbox("Select a machine learning algorithm:", ["AdaBoost", "Balanced Random Forest", "Decision Tree", "Easy Ensemble", "Gaussian Naïve Bayes","Gradient Boosting", "K-Nearest Neighbors", "Logistic Regression", "Random Forest",  "Stochastic Gradient Descent", "Support Vector"])
		
        if algorithm == "AdaBoost":
	        algorithm_import = "from sklearn.ensemble import AdaBoostClassifier"
	        algorithm_instance = "abc"
	        algorithm_class = "AdaBoostClassifier()"

        elif algorithm == "Balanced Random Forest":
	        algorithm_import = "from imblearn.ensemble import BalancedRandomForestClassifier"
	        algorithm_instance = "brfc"
	        algorithm_class = "BalancedRandomForestClassifier()"

        elif algorithm == "Decision Tree":
	        algorithm_import = "from sklearn import tree"
	        algorithm_instance = "dt"
	        algorithm_class = "tree.DecisionTreeClassifier()"

        elif algorithm == "Easy Ensemble":
	        algorithm_import = "from imblearn.ensemble import EasyEnsembleClassifier"
	        algorithm_instance = "eec"
	        algorithm_class = "EasyEnsembleClassifier()"

        elif algorithm == "Gaussian Naïve Bayes":
	        algorithm_import = "from sklearn.naive_bayes import GaussianNB"
	        algorithm_instance = "gnb"
	        algorithm_class = "GaussianNB()"

        elif algorithm == "Gradient Boosting":
	        algorithm_import = "from sklearn.ensemble import GradientBoostingClassifier"
	        algorithm_instance = "gbc"
	        algorithm_class = "GradientBoostingClassifier()"

        elif algorithm == "K-Nearest Neighbors":
	        algorithm_import = "from sklearn.neighbors import KNeighborsClassifier"
	        algorithm_instance = "knn"
	        algorithm_class = "KNeighborsClassifier()"

        elif algorithm == "Logistic Regression":
	        algorithm_import = "from sklearn.linear_model import LogisticRegression"
	        algorithm_instance = "lr"
	        algorithm_class = "LogisticRegression()"

        elif algorithm == "Random Forest":
	        algorithm_import = "from sklearn.ensemble import RandomForestClassifier"
	        algorithm_instance = "rfc"
	        algorithm_class = "RandomForestClassifier()"

        elif algorithm == "Support Vector":
	        algorithm_import = "from sklearn.svm import SVC"
	        algorithm_instance = "svm"
	        algorithm_class = "SVC()"

        elif algorithm == "Stochastic Gradient Descent":
	        algorithm_import = "from sklearn.linear_model import SGDClassifier"
	        algorithm_instance = "sgdc"
	        algorithm_class = "SGDClassifier()"
			
		#--------------------------------------------------------------------
