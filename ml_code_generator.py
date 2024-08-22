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
    Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( 📑 [Resume](https://resume-avijitc.streamlit.app/) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''',
    unsafe_allow_html=True)
st.info('**An easy-to-use, open-source application to generate python codes for machine learning algorithms**', icon="ℹ️")
#----------------------------------------
stats_expander = st.expander("**:blue[Instructions]**", expanded=False)
with stats_expander:
			
    st.write("1. Specify the variables in the parameters columns on the side bar")
    st.write("2. Copy the generated Python script to your clipboard")
    st.write("3. Paste the generated Python script on your IDE of preference")
    st.write("4. Run the Python script")
#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns((0.2,0.8))

with col1:

        st.subheader("Parameters", divider='blue')
        data_source = st.selectbox("**Select the file extension**", [".csv file", ".xlsx file"])
        if data_source == ".csv file":
	        data_source = "csv"
        else:
	        data_source = "excel" 
		
        #--------------------------------------------------------------------
		
        path = st.text_input("**Enter the input file path here**", "Desktop/")
		
		#--------------------------------------------------------------------
        
        st.divider()
        algorithm = st.selectbox("**Select a machine learning algorithm**", ["AdaBoost", "Balanced Random Forest", "Decision Tree", "Easy Ensemble", "Gaussian Naïve Bayes","Gradient Boosting", "K-Nearest Neighbors", "Logistic Regression", "Random Forest",  "Stochastic Gradient Descent", "Support Vector"])
		
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
		
        train_test_ratio = st.number_input("**Enter the percentage of the training set**", 0, max_value = 99, value = 70)
		
        #--------------------------------------------------------------------
		
        st.divider()
        scaling = st.selectbox("**Select a scalling algorithm**",["Max Abs Scaler", "Min Max Scaler", "min max scale", "Normalizer", "Power Transformer", "Quantile Transformer", "Robust Scaler", "Standard Scaler"])

        if scaling == "Standard Scaler":
	        scaling_technique_import = "from sklearn.preprocessing import StandardScaler"
	        scaling_class = "StandardScaler()"

        elif scaling == "Min Max Scaler":
	        scaling_technique_import = "from sklearn.preprocessing import MinMaxScaler"
	        scaling_class = "MinMaxScaler()"

        elif scaling == "min max scale":
	        scaling_technique_import = "from sklearn.preprocessing import minmax_scale"
	        scaling_class = "minmax_scale()"

        elif scaling == "Max Abs Scaler":
	        scaling_technique_import = "from sklearn.preprocessing import MaxAbsScaler"
	        scaling_class = "MaxAbsScaler()"

        elif scaling == "Robust Scaler":
	        scaling_technique_import = "from sklearn.preprocessing import RobustScaler"
	        scaling_class = "RobustScaler()"

        elif scaling == "Normalizer":
	        scaling_technique_import = "from sklearn.preprocessing import Normalizer"
	        scaling_class = "Normalizer()"

        elif scaling == "Quantile Transformer":
	        scaling_technique_import = "from sklearn.preprocessing import QuantileTransformer"
	        scaling_class = "QuantileTransformer()"

        elif scaling == "Power Transformer":
	        scaling_technique_import = "from sklearn.preprocessing import PowerTransformer"
	        scaling_class = "PowerTransformer()"
			
        #--------------------------------------------------------------------
		
        under_or_over = st.selectbox("**Select a resampling technique**", ["Oversampling", "Undersampling", "Combination"])

        if under_or_over == "Oversampling":
	        resampling = st.selectbox("**Select an oversampling technique**", ["ADASYN", "Borderline SMOTE", "Random Over Sampler","SMOTE", "SMOTEN", "SMOTENC"])

	        if resampling == "ADASYN":
		        resampling_import = "from imblearn.over_sampling import ADASYN"
		        resampling_instance = "adasyn"
		        resampling_class = "ADASYN()"

	        elif resampling == "Borderline SMOTE":
		        resampling_import = "from imblearn.over_sampling import BorderlineSMOTE"
		        resampling_instance = "bls"
		        resampling_class = "BorderlineSMOTE()"

	        elif resampling == "Random Over Sampler":
		        resampling_import = "from imblearn.over_sampling import RandomOverSampler"
		        resampling_instance = "ros"
		        resampling_class = "RandomOverSampler()"

	        elif resampling == "SMOTE":
		        resampling_import = "from imblearn.over_sampling import SMOTE"
		        resampling_instance = "smote"
		        resampling_class = "SMOTE()"

	        elif resampling == "SMOTEN":
		        resampling_import = "from imblearn.over_sampling import SMOTEN"
		        resampling_instance = "smoten"
		        resampling_class = "SMOTEN()"

	        elif resampling == "SMOTENC":
		        resampling_import = "from imblearn.over_sampling import SMOTENC"
		        resampling_instance = "smotenc"
		        resampling_class = "SMOTENC()"

        elif under_or_over == "Undersampling":
	        resampling = st.selectbox("**Select an undersampling technique**", ["All KNN" , "Cluster Centroids", "Condensed Nearest Neighbour", "Edited Nearest Neighbours", "Near Miss", "Neighbourhood Cleaning Rule", "One Sided Selection", "Random Under Sampler", "Repeated Edited Nearest Neighbours"])

	        if resampling == "All KNN":
		        resampling_import = "from imblearn.under_sampling import AllKNN"
		        resampling_instance = "akk"
		        resampling_class = "AllKNN()"

	        elif resampling == "Cluster Centroids":
		        resampling_import = "from imblearn.under_sampling import ClusterCentroids"
		        resampling_instance = "cc"
		        resampling_class = "ClusterCentroids()"

	        elif resampling == "Condensed Nearest Neighbour":
		        resampling_import = "from imblearn.under_sampling import CondensedNearestNeighbour"
		        resampling_instance = "cnn"
		        resampling_class = "CondensedNearestNeighbour()"

	        elif resampling == "Edited Nearest Neighbours":
		        resampling_import = "from imblearn.under_sampling import EditedNearestNeighbours"
		        resampling_instance = "enn"
		        resampling_class = "EditedNearestNeighbours"

	        elif resampling == "Near Miss":
		        resampling_import = "from imblearn.under_sampling import NearMiss"
		        resampling_instance = "nm1"
		        resampling_class = "NearMiss(version=1)"

	        elif resampling == "Neighbourhood Cleaning Rule":
		        resampling_import = "from imblearn.under_sampling import NeighbourhoodCleaningRule"
		        resampling_instance = "ncr"
		        resampling_class = "NeighbourhoodCleaningRule"

	        elif resampling == "One Sided Selection":
		        resampling_import = "from imblearn.under_sampling import OneSidedSelection"
		        resampling_instance = "oss"
		        resampling_class = "OneSidedSelection"

	        elif resampling == "Random Under Sampler":
		        resampling_import = "from imblearn.under_sampling import RandomUnderSampler"
		        resampling_instance = "rus"
		        resampling_class = "RandomUnderSampler()"

	        elif resampling == "Repeated Edited Nearest Neighbours":
		        resampling_import = "from imblearn.under_sampling import RepeatedEditedNearestNeighbours"
		        resampling_instance = "renn"
		        resampling_class = "RepeatedEditedNearestNeighbours()"

        elif under_or_over == "Combination":
	        resampling = st.selectbox("**Select a cmbination technique**",["SMOTEENN", "SMOTE Tomek"])

	        if resampling == "SMOTEENN":
		        resampling_import = "from imblearn.combine import SMOTEENN"
		        resampling_instance = "smoteenn"
		        resampling_class = "SMOTEENN()"

	        elif resampling == "SMOTE Tomek":
		        resampling_import = "from imblearn.combine import SMOTETomek"
		        resampling_instance = "smotetomek"
		        resampling_class = "SMOTETomek()"
				

with col2:
	
        st.subheader("Code:", divider='blue')
		
        st.code(

	            "# Import libraries and dependencies" +"\n"+ 
	            "import numpy as np" +"\n"+ 
	            "import pandas as pd" +"\n\n"+

	            "# ------------------------------ Data Set Loading ------------------------------" +"\n\n"+

	            "# Read data set" +"\n"+
	            "df = pd.read_" + data_source + "('" + path + "')" +"\n\n"+

	            "# Visualize data set" +"\n"+
	            "display(df.head())" +"\n\n"+ 

	            "# ------------------------------- Data Cleaning --------------------------------" +"\n\n"+

	            "# Remove null values" +"\n"+
	            "df.dropna(inplace = True)" +"\n\n"+

	            "# Specify the features columns" +"\n"+
	            "X = df.drop(columns = [df.columns[-1]])" +"\n\n"+

	            "# Specify the target column" +"\n"+
	            "y = df.iloc[:,-1]" +"\n\n"+

	            "# Transform non-numerical columns into binary-type columns" +"\n"+
	            "X = pd.get_dummies(X)" +"\n\n"+

	            "# ----------------------------- Data Preprocessing -----------------------------" +"\n\n"+

	            "# Import train_test_split class" +"\n"+ 
	            "from sklearn.model_selection import train_test_split" +"\n\n"+ 

	            "# Divide data set into traning and testing subsets" +"\n"+ 
	            "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = " + str(round(train_test_ratio/100,2)) + ")" +"\n\n"+

	            "# Import data scaling technique class" +"\n"+ 
	            scaling_technique_import +"\n\n"+

	            "# Instantiate data scaler" +"\n"+ 
	            "scaler = " + scaling_class +"\n\n"+ ""

	            "# Fit the Scaler with the training data" +"\n"+ 
	            "X_scaler = scaler.fit(X_train)" +"\n\n"+

	            "# Scale the training and testing data" +"\n"+ 
	            "X_train_scaled = X_scaler.transform(X_train)" +"\n"+ 
	            "X_test_scaled = X_scaler.transform(X_test)" +"\n\n"+

	            "# ------------------------------ Data Resampling ------------------------------" +"\n\n"+

	            "# Import data resampling class" +"\n"+ 
	            resampling_import +"\n\n"+

	            "# Instantiate data resampler technique" +"\n"+ 
	            resampling_instance + " = " + resampling_class +"\n\n"+

	            "# Resample training sets" +"\n"+ 
	            "X_resampled, y_resampled = " + resampling_instance + ".fit_resample(X_train_scaled, y_train)" +"\n\n"+

	            "# ------------------------------- Model Building -------------------------------" +"\n\n"+ 

	            "# Import machine learning model class" +"\n"+ 
	            algorithm_import +"\n\n"+ 

	            "# Instantiate machine learning model" +"\n"+ 
	            algorithm_instance + " = " + algorithm_class +"\n\n"+

	            "# Fit the machine learning model with the training data" +"\n"+
	            algorithm_instance + '.fit(X_resampled, y_resampled)' +"\n\n"+

	            "# Make predictions using the testing data" +"\n"+ 
	            "y_pred = " + algorithm_instance + '.predict(X_test_scaled)' +"\n\n"+ 

	            "# ------------------------------ Model Evaluation ------------------------------" +"\n\n"+

	            "# Calculate balanced accuracy scrore" +"\n"+ 
	            "from sklearn.metrics import balanced_accuracy_score" +"\n"+
	            "print(balanced_accuracy_score(y_test, y_pred))" +"\n\n"+

	            "# Display the confusion matrix" +"\n"+
	            "from sklearn.metrics import confusion_matrix" +"\n"+
	            "print(confusion_matrix(y_test, y_pred))" +"\n\n"+

	            "# Display the imbalanced classification report" +"\n"+
	            "from imblearn.metrics import classification_report_imbalanced" +"\n"+
	            "print(classification_report_imbalanced(y_test, y_pred))"

	            )
