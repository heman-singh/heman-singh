import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

st.set_page_config(page_title= "Upload Data")

upload = st.file_uploader("Upload your excel file")
if upload is not None:
    df = pd.read_csv(upload)
    with st.expander("Data"):
        st.write(df)
    
    df_new = df.drop_duplicates()
    
    tabs_list = ["Shape", "Size", "Null Values", "Duplictes Values", "Describe"]    
    tab_shape, tab_size, tab_null, tab_dplct, tab_describe = st.tabs(tabs_list)    
    with tab_shape:
        st.write("Shape of Data (Row, Column): " , df.shape)
    with tab_size:
        st.write("Size of Data: ", df.size)
    
    with tab_null:
        null_num=df.isnull().sum()
        null = pd.DataFrame()
        null['feature'] = null_num.index
        null['null values'] = np.array(null_num.values)      
        st.write("Null values in data: " , null)
        
        n_row = df.shape[0]*60/100
       
         
        null_val = null[null['null values']>0] 
        
        null_fill=null_val[null_val['null values'] <= n_row ]
        null_drop = null_val[null_val['null values'] > n_row]
        
        list_fill = [null_fill["null values"]]
        list_drop = [null_drop["null values"]]
        for a in null_fill['null values']:
            if a > 0:
                for i in null_fill['feature']:
                    df_new[i].fillna(df[i].median(), inplace = True)
                
        for b in null_drop['null values']:
            if b > 0:
                for k in null_drop['feature']:
                    df_new.drop([i], axis=1, inplace= True)
        
        
        st.write("Now there is no null values in data", df_new.isnull().sum())
        with st.expander("New data "):
            st.write(df_new)
            
        dup = df_new.duplicated().sum()
        st.write("Number of duplicate rows in data: " , dup)
        if dup == 0:
            st.text("Data has no duplicate value.")
        else:
            st.write("Duplicate values has been removed.")
            with st.expander("New Data"):
                df_new = df_new.drop_duplicates()
                st.write(df_new, "Shape of New Data (Row, Column):",df_new.shape)
        
        
    with tab_describe:
        st.write(df_new.describe())
        
    
    
    st.header("ML models")
    
    with st.expander("Random Forest"):
        
    
        
        st.header("Random Forest Classifier")
    
        dependent_features_rf = st.selectbox("Dependent Features:", df_new.columns)
        ind_feature_rf = st.multiselect("Independent Feature: ", df_new.drop([dependent_features_rf], axis=1).columns)
        
        if len(ind_feature_rf) == 0:
            st.write("Please select the Independent Feature.")
        else:
            
            X = df_new[ind_feature_rf]
            y = df_new[dependent_features_rf]
            
            X_train , X_test, y_train, y_test = train_test_split(X , y , test_size=0.2, shuffle=False)
            scaler = StandardScaler()
            X_train =scaler.fit_transform(X_train)
            X_test =scaler.transform(X_test)
            
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            
            y_pred_rf = rf.predict(X_test)
            
            
            st.subheader("Correlation Plot")
            corr = df_new.corr()
            corr2 = corr[dependent_features_rf]
            corr_data = pd.DataFrame()
            corr_data["Features"] = corr2.index
            corr_data["Correlation"] = corr2.values
            st.bar_chart(corr_data, x = 'Features', y = 'Correlation')
        
            st.subheader("Model Evaluation")
            tab_con_rf, tab_acc_rf, tab_pre_rf, tab_rec_rf = st.tabs(["Confusion Metrix","Accuracy", "Precision", "Racall"])
            with tab_con_rf:
                st.write("Confusion Metrix : ", confusion_matrix(y_test, y_pred_rf))
            with tab_acc_rf:
                st.write("Accurac of model :", accuracy_score(y_test, y_pred_rf)*100)
            with tab_pre_rf:
                st.write("Precision of model: ", precision_score(y_test, y_pred_rf, average = 'micro')*100)
            with tab_rec_rf:
                st.write("Recall score of model: ", recall_score(y_test, y_pred_rf, average = 'micro')*100)

            st.subheader("Feature Importance")
            chart_rf = pd.DataFrame()
            chart_rf['Features'] = X.columns
            chart_rf['Feature_Importance'] = rf.feature_importances_
   
            st.bar_chart(chart_rf, x = 'Features', y= 'Feature_Importance')

            
            with st.sidebar:  # sidebar
                st.header("Prediction")
                upload_data_pred , indi_pred = st.tabs(["Upload a file ", "Individual Prediction"])
                with upload_data_pred:
                    pred_file = st.file_uploader("Upload your file for prediction.")
                    if pred_file is not None:
                        pred_data = pd.read_excel(pred_file)
                        predicton = rf.predict(pred_data)
                    
                        a = list()
                        b = list()
                        for i in pred_data.columns:
                            a.append(i)
                        for k in X.columns:    
                            b.append(k)
                    
                        if a != b :
                            st.write("Try again")
                        else:
                            st.write(predicton)
                with indi_pred:
                    i_list = []
                    for i in enumerate(X.columns):
                        i = st.text_input('{}'.format(i).upper() ,0 )
                        i_list.append(i)
                        result= ""
                    
                    
                    if st.button("Predict"):
                        i_list = np.array(i_list).reshape(-1,1)
                        st.write(i_list.T)
                        result = rf.predict(i_list.T)
                    st.success('The output is {}'.format(result))
                
                
