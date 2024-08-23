import streamlit as st

# Set page configuration
st.set_page_config(page_title="ML Code Generator | v0.1",
                   layout="wide",
                   page_icon="💻",            
                   initial_sidebar_state="collapsed")

# Title and description
st.title(f""":rainbow[ML Code Generator]""")
st.markdown(
    '''
    Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( 📑 [Resume](https://resume-avijitc.streamlit.app/) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''',
    unsafe_allow_html=True)
st.info('**An easy-to-use, open-source application to generate python codes for machine learning algorithms**', icon="ℹ️")

# Instructions
stats_expander = st.expander("**:blue[Instructions]**", expanded=False)
with stats_expander:
    st.write("1. Specify the variables in the parameters columns on the side bar")
    st.write("2. Copy the generated Python script to your clipboard")
    st.write("3. Paste the generated Python script on your IDE of preference")
    st.write("4. Run the Python script")

# Main app
col1, col2 = st.columns((0.2, 0.8))

with col1:
    st.subheader("Parameters", divider='blue')
    data_source = st.selectbox("**Select the file extension**", [".csv file", ".xlsx file"])
    if data_source == ".csv file":
        data_source = "csv"
    else:
        data_source = "excel"

    path = st.text_input("**Enter the input file path here**", "Desktop/")

    # Dropdown to select between classification and regression
    task_type = st.selectbox("**Select the type of task**", ["Classification", "Regression"])

    if task_type == "Classification":
        algorithm = st.selectbox("**Select a machine learning algorithm**", ["AdaBoost", "Balanced Random Forest", "Decision Tree", "Easy Ensemble", "Gaussian Naïve Bayes", "Gradient Boosting", "K-Nearest Neighbors", "Logistic Regression", "Random Forest",  "Stochastic Gradient Descent", "Support Vector"])

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

    elif task_type == "Regression":
        algorithm = st.selectbox("**Select a machine learning algorithm**", ["Linear Regression", "Ridge", "Lasso", "Elastic Net", "Random Forest Regressor", "Gradient Boosting Regressor", "Support Vector Regressor"])

        if algorithm == "Linear Regression":
            algorithm_import = "from sklearn.linear_model import LinearRegression"
            algorithm_instance = "lr"
            algorithm_class = "LinearRegression()"

        elif algorithm == "Ridge":
            algorithm_import = "from sklearn.linear_model import Ridge"
            algorithm_instance = "ridge"
            algorithm_class = "Ridge()"

        elif algorithm == "Lasso":
            algorithm_import = "from sklearn.linear_model import Lasso"
            algorithm_instance = "lasso"
            algorithm_class = "Lasso()"

        elif algorithm == "Elastic Net":
            algorithm_import = "from sklearn.linear_model import ElasticNet"
            algorithm_instance = "en"
            algorithm_class = "ElasticNet()"

        elif algorithm == "Random Forest Regressor":
            algorithm_import = "from sklearn.ensemble import RandomForestRegressor"
            algorithm_instance = "rfr"
            algorithm_class = "RandomForestRegressor()"

        elif algorithm == "Gradient Boosting Regressor":
            algorithm_import = "from sklearn.ensemble import GradientBoostingRegressor"
            algorithm_instance = "gbr"
            algorithm_class = "GradientBoostingRegressor()"

        elif algorithm == "Support Vector Regressor":
            algorithm_import = "from sklearn.svm import SVR"
            algorithm_instance = "svr"
            algorithm_class = "SVR()"

    train_test_ratio = st.number_input("**Enter the percentage of the training set**", 0, max_value=99, value=70)

    # Dropdown to select resampling method
    resampling_method = st.selectbox("**Select a resampling method**", ["None", "Random Oversampler", "SMOTE", "ADASYN", "Random Undersampler", "Tomek Links"])

    st.divider()
    scaling = st.selectbox("**Select a scaling algorithm**", ["Max Abs Scaler", "Min Max Scaler", "Normalizer", "Power Transformer", "Quantile Transformer", "Robust Scaler", "Standard Scaler"])

    if scaling == "Standard Scaler":
        scaling_technique_import = "from sklearn.preprocessing import StandardScaler"
        scaling_class = "StandardScaler()"

    elif scaling == "Min Max Scaler":
        scaling_technique_import = "from sklearn.preprocessing import MinMaxScaler"
        scaling_class = "MinMaxScaler()"

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

with col2:
    st.subheader("Code:", divider='blue')

    code = (
        "# Import libraries and dependencies\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"

        "# ------------------------------ Data Set Loading ------------------------------\n\n"

        "# Read data set\n"
        f"df = pd.read_{data_source}('{path}')\n\n"

        "# Visualize data set\n"
        "display(df.head())\n\n"

        "# ------------------------------- Data Cleaning --------------------------------\n\n"

        "# Remove null values\n"
        "df.dropna(inplace=True)\n\n"

        "# Specify the features columns\n"
        "X = df.drop(columns=[df.columns[-1]])\n\n"

        "# Specify the target column\n"
        "y = df.iloc[:, -1]\n\n"

        "# Transform non-numerical columns into binary-type columns\n"
        "X = pd.get_dummies(X)\n\n"

        "# ----------------------------- Data Preprocessing -----------------------------\n\n"

        "# Import train_test_split class\n"
        "from sklearn.model_selection import train_test_split\n\n"

        "# Divide data set into training and testing subsets\n"
        f"X_train, X_test, y_train, y_test = train_test_split(X, y, train_size={str(round(train_test_ratio / 100, 2))})\n\n"
    )

    if resampling_method != "None":
        if resampling_method == "Random Oversampler":
            code += (
                "# Import RandomOverSampler\n"
                "from imblearn.over_sampling import RandomOverSampler\n\n"
                "# Resample the training data using RandomOverSampler\n"
                "ros = RandomOverSampler()\n"
                "X_train, y_train = ros.fit_resample(X_train, y_train)\n\n"
            )

        elif resampling_method == "SMOTE":
            code += (
                "# Import SMOTE\n"
                "from imblearn.over_sampling import SMOTE\n\n"
                "# Resample the training data using SMOTE\n"
                "smote = SMOTE()\n"
                "X_train, y_train = smote.fit_resample(X_train, y_train)\n\n"
            )

        elif resampling_method == "ADASYN":
            code += (
                "# Import ADASYN\n"
                "from imblearn.over_sampling import ADASYN\n\n"
                "# Resample the training data using ADASYN\n"
                "adasyn = ADASYN()\n"
                "X_train, y_train = adasyn.fit_resample(X_train, y_train)\n\n"
            )

        elif resampling_method == "Random Undersampler":
            code += (
                "# Import RandomUnderSampler\n"
                "from imblearn.under_sampling import RandomUnderSampler\n\n"
                "# Resample the training data using RandomUnderSampler\n"
                "rus = RandomUnderSampler()\n"
                "X_train, y_train = rus.fit_resample(X_train, y_train)\n\n"
            )

        elif resampling_method == "Tomek Links":
            code += (
                "# Import TomekLinks\n"
                "from imblearn.under_sampling import TomekLinks\n\n"
                "# Resample the training data using TomekLinks\n"
                "tl = TomekLinks()\n"
                "X_train, y_train = tl.fit_resample(X_train, y_train)\n\n"
            )

    code += (
        "# Scale data\n"
        f"{scaling_technique_import}\n"
        f"scaler = {scaling_class}\n"
        "X_train = scaler.fit_transform(X_train)\n"
        "X_test = scaler.transform(X_test)\n\n"

        "# ----------------------------- Model Development ------------------------------\n\n"

        "# Import model classes\n"
        f"{algorithm_import}\n\n"

        "# Instantiate model classes\n"
        f"{algorithm_instance} = {algorithm_class}\n\n"

        "# Fit model on training data\n"
        f"{algorithm_instance}.fit(X_train, y_train)\n\n"
    )

    if task_type == "Classification":
        code += (
            "# Generate predictions on testing data\n"
            f"y_pred = {algorithm_instance}.predict(X_test)\n\n"

            "# ----------------------------- Model Evaluation -------------------------------\n\n"

            "# Generate classification report and confusion matrix\n"
            "from sklearn.metrics import classification_report, confusion_matrix\n"
            "print(confusion_matrix(y_test, y_pred))\n"
            "print(classification_report(y_test, y_pred))\n"
        )

    elif task_type == "Regression":
        code += (
            "# Generate predictions on testing data\n"
            f"y_pred = {algorithm_instance}.predict(X_test)\n\n"

            "# ----------------------------- Model Evaluation -------------------------------\n\n"

            "# Evaluate model performance using R2 score and Mean Squared Error\n"
            "from sklearn.metrics import r2_score, mean_squared_error\n"
            "print(f'R2 Score: {r2_score(y_test, y_pred)}')\n"
            "print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')\n"
        )

    st.code(code, language='python')
