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
from datetime import datetime
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="ML Code Generator | v0.4",
                   layout="wide",
                   page_icon="💻",            
                   initial_sidebar_state="collapsed")
#---------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .version-badge {
        text-align: center;
        display: inline-block;
        background: linear-gradient(120deg, #0056b3, #0d4a96);
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 1.15rem;
        margin-top: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    </style>
    <div style="text-align: center;">
        <div class="title-large">Machine Learning (ML) Code Generator</div>
        <div class="version-badge"> Play with Code | v0.4 </div>
    </div>
    """,
    unsafe_allow_html=True)
#---------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2026 | Created by : <span class="highlight">Avijit Chakraborty</span> <a href="mailto:avijit.mba18@gmail.com"> 📩 </a> | <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span> </p>
    </div>
    """,
    unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### CSS 
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown(
            """
            <style>
                .centered-info {
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 15px;
                color: #007BFF; 
                padding: 5px;
                background-color: #FFFFFF; 
                border-radius: 20px;
                border: 1px solid #007BFF;
                margin-top: 5px;
                margin-bottom : 5px;
                }
            </style>
            """,unsafe_allow_html=True,)

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------
#st.markdown('<div class="centered-info"><span style="margin-left: 10px;">An easy-to-use, open-source application to generate python codes for machine learning algorithms</span></div>',unsafe_allow_html=True,)

# -----------------------------------------------------------------------------
# Helper Dictionaries for Mappings
# -----------------------------------------------------------------------------
ALGO_MAP_CLASSIFICATION = {
        "AdaBoost": {"import": "from sklearn.ensemble import AdaBoostClassifier", "class": "AdaBoostClassifier()", "inst": "ada"},
        "Balanced Random Forest": {"import": "from imblearn.ensemble import BalancedRandomForestClassifier", "class": "BalancedRandomForestClassifier()", "inst": "brf"},
        "Decision Tree": {"import": "from sklearn.tree import DecisionTreeClassifier", "class": "DecisionTreeClassifier()", "inst": "dt"},
        "Gaussian Naïve Bayes": {"import": "from sklearn.naive_bayes import GaussianNB", "class": "GaussianNB()", "inst": "gnb"},
        "Gradient Boosting": {"import": "from sklearn.ensemble import GradientBoostingClassifier", "class": "GradientBoostingClassifier()", "inst": "gb"},
        "K-Nearest Neighbors": {"import": "from sklearn.neighbors import KNeighborsClassifier", "class": "KNeighborsClassifier()", "inst": "knn"},
        "Logistic Regression": {"import": "from sklearn.linear_model import LogisticRegression", "class": "LogisticRegression()", "inst": "lr"},
        "Random Forest": {"import": "from sklearn.ensemble import RandomForestClassifier", "class": "RandomForestClassifier()", "inst": "rf"},
        "SVM": {"import": "from sklearn.svm import SVC", "class": "SVC()", "inst": "svm"},
        "SGD Classifier": {"import": "from sklearn.linear_model import SGDClassifier", "class": "SGDClassifier()", "inst": "sgd"},
    }

ALGO_MAP_REGRESSION = {
        "Linear Regression": {"import": "from sklearn.linear_model import LinearRegression", "class": "LinearRegression()", "inst": "lr"},
        "Ridge": {"import": "from sklearn.linear_model import Ridge", "class": "Ridge()", "inst": "ridge"},
        "Lasso": {"import": "from sklearn.linear_model import Lasso", "class": "Lasso()", "inst": "lasso"},
        "Elastic Net": {"import": "from sklearn.linear_model import ElasticNet", "class": "ElasticNet()", "inst": "en"},
        "Random Forest Regressor": {"import": "from sklearn.ensemble import RandomForestRegressor", "class": "RandomForestRegressor()", "inst": "rfr"},
        "Gradient Boosting Regressor": {"import": "from sklearn.ensemble import GradientBoostingRegressor", "class": "GradientBoostingRegressor()", "inst": "gbr"},
        "SVR": {"import": "from sklearn.svm import SVR", "class": "SVR()", "inst": "svr"},
    }

SCALING_MAP = {
        "Standard Scaler": {"import": "from sklearn.preprocessing import StandardScaler", "class": "StandardScaler()"},
        "Min Max Scaler": {"import": "from sklearn.preprocessing import MinMaxScaler", "class": "MinMaxScaler()"},
        "Max Abs Scaler": {"import": "from sklearn.preprocessing import MaxAbsScaler", "class": "MaxAbsScaler()"},
        "Robust Scaler": {"import": "from sklearn.preprocessing import RobustScaler", "class": "RobustScaler()"},
        "Normalizer": {"import": "from sklearn.preprocessing import Normalizer", "class": "Normalizer()"},
        "Quantile Transformer": {"import": "from sklearn.preprocessing import QuantileTransformer", "class": "QuantileTransformer()"},
        "Power Transformer": {"import": "from sklearn.preprocessing import PowerTransformer", "class": "PowerTransformer()"},
    }

FS_MAP = {
        "SelectKBest": "SelectKBest",
        "Recursive Feature Elimination": "RFE",
        "Feature Importance": "Importance",
        "PCA": "PCA",
    }

RESAMPLE_MAP = {
        "Random Oversampler": {"type": "over", "import": "from imblearn.over_sampling import RandomOverSampler", "class": "RandomOverSampler()"},
        "SMOTE": {"type": "over", "import": "from imblearn.over_sampling import SMOTE", "class": "SMOTE()"},
        "ADASYN": {"type": "over", "import": "from imblearn.over_sampling import ADASYN", "class": "ADASYN()"},
        "Random Undersampler": {"type": "under", "import": "from imblearn.under_sampling import RandomUnderSampler", "class": "RandomUnderSampler()"},
        "Tomek Links": {"type": "under", "import": "from imblearn.under_sampling import TomekLinks", "class": "TomekLinks()"},
    }    
#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------
st.divider()
col1, col2 = st.columns((0.2, 0.8))
with col1:
    
    #st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Parameters</span></div>',unsafe_allow_html=True,)
    #--------------------------------------------------------------------

    with st.container(border=True):
        data_source = st.selectbox("File Extension", [".csv file", ".xlsx file"])
        data_source = "csv" if data_source == ".csv file" else "excel"
        path = st.text_input("Input File Path", "data/dataset.csv")
        
        st.write("")
        task_type = st.selectbox("Task Type", ["Classification", "Regression"])
        
        st.write("")
        train_ratio = st.slider("Training Set Percentage", 10, 90, 70)

        st.write("")
        if task_type == "Classification":
            selected_algos = st.multiselect("Select Algorithms", list(ALGO_MAP_CLASSIFICATION.keys()), default=["Random Forest", "Logistic Regression"])
            algo_map = ALGO_MAP_CLASSIFICATION
        else:
            selected_algos = st.multiselect("Select Algorithms", list(ALGO_MAP_REGRESSION.keys()), default=["Linear Regression", "Random Forest Regressor"])
            algo_map = ALGO_MAP_REGRESSION      
        
        selected_fs = st.multiselect("Select Methods", list(FS_MAP.keys()),)
        k_features = st.number_input("K Features (for SelectKBest/RFE)", min_value=1, value=10)
                
        st.write("")
        selected_scaling = st.selectbox("Scaling Method", list(SCALING_MAP.keys()), index=6) # Default Standard
        selected_resample = st.selectbox("Resampling Method", ["None"] + list(RESAMPLE_MAP.keys()))
                
        st.write("")
        include_cv = st.checkbox("Include Cross-Validation", value=True)
        include_hp = st.checkbox("Include Hyperparameter Tuning", value=False)
        include_shap = st.checkbox("Include SHAP Analysis", value=True) 
        
        st.info("The generated code will automatically compare all selected Algorithms x Feature Selection combinations and output a results DataFrame.")
                           
        with col2:
            
            if st.button("Generate Code", type="primary"):
                    
                    #st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Code</span></div>',unsafe_allow_html=True,)
                    #st.caption("Generated by ML Code Generator Pro | Ensure `scikit-learn`, `pandas`, `imblearn`, and `shap` are installed.")    
                    
                    def generate_code():

                        code = []
                        
                        # Header Comments
                        code.append("#" + "="*80)
                        code.append("# ML Pipeline - Auto Generated Code")
                        code.append(f"# Task Type: {task_type}")
                        code.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        code.append(f"# Algorithms: {', '.join(selected_algos)}")
                        code.append(f"# Feature Selection: {', '.join(selected_fs)}")
                        code.append("#" + "="*80)
                        code.append("")
                        
                        # Import libraries and dependencies
                        code.append("# Import libraries and dependencies")
                        code.append("import numpy as np")
                        code.append("import pandas as pd")
                        code.append("import warnings")
                        code.append("warnings.filterwarnings('ignore')")
                        code.append("")
                        
                        # --- Data Loading ---
                        code.append("# ------------------------------ Data Set Loading ------------------------------")
                        read_func = "read_csv" if data_source == "csv" else "read_excel"
                        code.append(f"df = pd.{read_func}('{path}')")
                        code.append("print(f'Data loaded: {df.shape}')")
                        code.append("")
                        
                        # --- Data Cleaning ---
                        code.append("# ------------------------------- Data Cleaning --------------------------------")
                        code.append("df.dropna(inplace=True)")
                        code.append("X = df.iloc[:, :-1]")
                        code.append("y = df.iloc[:, -1]")
                        code.append("X = pd.get_dummies(X, drop_first=True)")
                        code.append("")
                        
                        # --- Outliers ---
                        code.append("# ----------------------------- Handling Outliers -----------------------------")
                        code.append("Q1 = X.quantile(0.25)")
                        code.append("Q3 = X.quantile(0.75)")
                        code.append("IQR = Q3 - Q1")
                        code.append("X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]")
                        code.append("y = y[X.index]")
                        code.append("")
                        
                        # --- Imports for Algorithms & Scaling ---
                        code.append("# ----------------------------- Imports ---------------------------------------")
                        imports_added = set()
                        for algo in selected_algos:
                            imp = algo_map[algo]["import"]
                            if imp not in imports_added:
                                code.append(imp)
                                imports_added.add(imp)
                        
                        sc_imp = SCALING_MAP[selected_scaling]["import"]
                        code.append(sc_imp)
                        
                        if selected_resample != "None":
                            code.append(RESAMPLE_MAP[selected_resample]["import"])
                        
                        code.append("from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV")
                        if task_type == "Classification":
                            code.append("from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report")
                        else:
                            code.append("from sklearn.metrics import r2_score, mean_squared_error")
                        code.append("")
                        
                        # --- Train Test Split ---
                        code.append("# ------------------------- Train-Test Split -----------------------------")
                        test_size = round((100 - train_ratio) / 100, 2)
                        code.append(f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)")
                        code.append("")
                        
                        # --- Resampling ---
                        if selected_resample != "None":
                            code.append("# ----------------------------- Resampling --------------------------------")
                            rs_class = RESAMPLE_MAP[selected_resample]["class"]
                            code.append(f"resampler = {rs_class}")
                            code.append("X_train, y_train = resampler.fit_resample(X_train, y_train)")
                            code.append("")
                        
                        # --- Scaling ---
                        code.append("# ----------------------------- Scaling -----------------------------------")
                        sc_class = SCALING_MAP[selected_scaling]["class"]
                        code.append(f"scaler = {sc_class}")
                        code.append("X_train = scaler.fit_transform(X_train)")
                        code.append("X_test = scaler.transform(X_test)")
                        code.append("")
                        
                        # --- Comparison Logic Setup ---
                        code.append("# ----------------------- Pipeline Comparison -----------------------------")
                        code.append("results = []")
                        code.append("")
                        
                        # Handle Feature Selection "None" case
                        fs_list = selected_fs if "None" not in selected_fs else ["None"]
                        
                        code.append("from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE")
                        code.append("from sklearn.decomposition import PCA")
                        code.append("")
                        
                        code.append("# Define Feature Selection Functions")
                        code.append(f"def apply_fs(method, X_tr, X_te, y_tr, k={k_features}):")
                        code.append("    if method == 'None':")
                        code.append("        return X_tr, X_te")
                        code.append("    elif method == 'SelectKBest':")
                        score_func = "f_classif" if task_type == "Classification" else "f_regression"
                        code.append(f"        selector = SelectKBest(score_func={score_func}, k=k)")
                        code.append("        X_tr = selector.fit_transform(X_tr, y_tr)")
                        code.append("        X_te = selector.transform(X_te)")
                        code.append("        return X_tr, X_te")
                        code.append("    elif method == 'RFE':")
                        code.append(f"        estimator = {list(algo_map.values())[0]['class']}") 
                        code.append("        selector = RFE(estimator=estimator, n_features_to_select=k)")
                        code.append("        X_tr = selector.fit_transform(X_tr, y_tr)")
                        code.append("        X_te = selector.transform(X_te)")
                        code.append("        return X_tr, X_te")
                        code.append("    elif method == 'PCA':")
                        code.append("        pca = PCA(n_components=k)")
                        code.append("        X_tr = pca.fit_transform(X_tr)")
                        code.append("        X_te = pca.transform(X_te)")
                        code.append("        return X_tr, X_te")
                        code.append("    return X_tr, X_te")
                        code.append("")
                        
                        code.append("# Main Comparison Loop")
                        code.append(f"fs_methods = {fs_list}")
                        
                        # Build Algo Instantiation String
                        algo_instantiations = []
                        for algo in selected_algos:
                            cls = algo_map[algo]["class"]
                            algo_instantiations.append(f"'{algo}': {cls}")
                        
                        code.append("algorithms = {" + ", ".join(algo_instantiations) + "}")
                        code.append("")
                        
                        code.append("for fs in fs_methods:")
                        code.append("    # Apply Feature Selection")
                        code.append(f"    X_train_fs, X_test_fs = apply_fs(fs, X_train.copy(), X_test.copy(), y_train.copy(), k={k_features})")
                        code.append("")
                        code.append("    for name, model in algorithms.items():")
                        code.append("        try:")
                        code.append("            # Fit")
                        code.append("            model.fit(X_train_fs, y_train)")
                        code.append("")
                        code.append("            # Predict")
                        code.append("            y_pred = model.predict(X_test_fs)")
                        code.append("")
                        code.append("            # Evaluate")
                        if task_type == "Classification":
                            code.append("            acc = accuracy_score(y_test, y_pred)")
                            code.append("            f1 = f1_score(y_test, y_pred, average='weighted')")
                            code.append("            score = acc")
                            code.append("            metric_name = 'Accuracy'")
                        else:
                            code.append("            r2 = r2_score(y_test, y_pred)")
                            code.append("            mse = mean_squared_error(y_test, y_pred)")
                            code.append("            score = r2")
                            code.append("            metric_name = 'R2'")
                        
                        code.append("")
                        if include_cv:
                            code.append("            # Cross Validation")
                            code.append("            cv_scores = cross_val_score(model, X_train_fs, y_train, cv=5)")
                            code.append("            cv_mean = cv_scores.mean()")
                        else:
                            code.append("            cv_mean = 0.0")
                            
                        code.append("")
                        code.append("            results.append({")
                        code.append("                'Feature_Selection': fs,")
                        code.append("                'Algorithm': name,")
                        if task_type == "Classification":
                            code.append("                'Accuracy': score,")
                        else:
                            code.append("                'R2': score,")
                        code.append("                'CV_Mean': cv_mean")
                        code.append("            })")
                        code.append("        except Exception as e:")
                        code.append("            print(f'Error with {name} + {fs}: {e}')")
                        code.append("")
                        
                        # Results DataFrame
                        code.append("# ----------------------------- Results Table -------------------------------")
                        code.append("df_results = pd.DataFrame(results)")
                        code.append("print('\\n' + '='*80)")
                        code.append("print('MODEL COMPARISON RESULTS')")
                        code.append("print('='*80)")
                        if task_type == "Classification":
                            code.append("print(df_results.sort_values(by='Accuracy', ascending=False))")
                            code.append("print(f\"\\nBest Model: {df_results.loc[df_results['Accuracy'].idxmax(), 'Algorithm']} with {df_results['Accuracy'].max():.4f} Accuracy\")")
                        else:
                            code.append("print(df_results.sort_values(by='R2', ascending=False))")
                            code.append("print(f\"\\nBest Model: {df_results.loc[df_results['R2'].idxmax(), 'Algorithm']} with {df_results['R2'].max():.4f} R2 Score\")")
                        code.append("")
                        
                        # Save Results to CSV
                        code.append("# Save results to CSV")
                        code.append("df_results.to_csv('model_comparison_results.csv', index=False)")
                        code.append("print('\\nResults saved to model_comparison_results.csv')")
                        code.append("")
                        
                        # Hyperparameter Tuning (Optional - Applied to Best Model)
                        if include_hp:
                            code.append("# ------------------------- Hyperparameter Tuning -------------------------")
                            code.append("# Tuning the best performing model from the list above")
                            code.append("best_algo_name = df_results.iloc[0]['Algorithm']")
                            code.append("best_model = algorithms[best_algo_name]")
                            code.append("param_grid = { 'n_estimators': [50, 100], 'max_depth': [None, 10] }")
                            code.append("grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=3, n_jobs=-1)")
                            code.append("grid_search.fit(X_train_fs, y_train)")
                            code.append("print(f'Best Params: {grid_search.best_params_}')")
                            code.append("")

                        # SHAP Analysis (Optional)
                        if include_shap:
                            code.append("# ----------------------------- SHAP Analysis -------------------------------")
                            code.append("try:")
                            code.append("    import shap")
                            code.append("    import matplotlib.pyplot as plt")
                            code.append("    best_model.fit(X_train_fs, y_train)")
                            code.append("    explainer = shap.Explainer(best_model, X_train_fs)")
                            code.append("    shap_values = explainer(X_test_fs[:100])")
                            code.append("    shap.summary_plot(shap_values, X_test_fs[:100])")
                            code.append("    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')")
                            code.append("    print('SHAP plot saved to shap_summary.png')")
                            code.append("except Exception as e:")
                            code.append("    print('SHAP analysis skipped due to error:', e)")
                            code.append("")
                        
                        code.append("#" + "="*80)
                        code.append("# End of Generated Code")
                        code.append("#" + "="*80)
                        
                        return "\n".join(code)

                        # Display in Streamlit
                        #st.code("\n".join(code), language="python")
                        
                    with st.container(border=True):
                        with st.spinner("Generating code..."):
                                st.session_state.generated_code = generate_code()
                                st.session_state.code_generated = True

                        with st.popover("**:red[How to use and implement the code]**", disabled=False, use_container_width=True):    
                            
                            st.info("The generated code will automatically compare all selected Algorithms x Feature Selection combinations and output a results DataFrame.")
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.markdown("""
                                ##### 📋 What's Included:
                                - ✅ Data Loading & Cleaning
                                - ✅ Outlier Detection (IQR)
                                - ✅ Feature Selection Methods
                                - ✅ Train-Test Split
                                - ✅ Resampling (if selected)
                                - ✅ Feature Scaling
                                - ✅ Model Comparison Loop
                                - ✅ Performance Metrics
                                - ✅ Cross-Validation (optional)
                                - ✅ Hyperparameter Tuning (optional)
                                - ✅ SHAP Analysis (optional)
                                """)
                            
                            with col_info2:
                                st.markdown("""
                                ##### 📦 Required Packages:
                                ```bash
                                pip install pandas numpy scikit-learn
                                pip install imblearn shap matplotlib
                                ```
                                
                                ##### 🚀 How to Run:
                                1. Download the .py file
                                2. Update the file path if needed
                                3. Run: `python ml_pipeline_*.py`
                                4. Check `model_comparison_results.csv`
                                """)
                                
                        if st.session_state.code_generated and st.session_state.generated_code:
                            st.code(st.session_state.generated_code, language="python")
                                                
                        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                        with col_dl2:
                            # Create filename with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"ml_pipeline_{task_type.lower()}_{timestamp}.py"
                            
                            st.download_button(label="📥 Download Code as .py File",data=st.session_state.generated_code,file_name=filename,mime="text/x-python",use_container_width=True,type="primary")

            else:
                st.error("Please run the 'Generate Code' Button.")
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
 
