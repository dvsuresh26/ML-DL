import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split

#imports for profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# import pandas_profiling
#from pandas_profiling import ProfileReport
#from scipy import interp
import h2o
from h2o.automl import H2OAutoML
with st.sidebar:
    st.title("Autostream ML")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This application will allow you to build an automated ML pipeline")


if os.path.exists("source_data.csv"):
    df=pd.read_csv("source_data.csv",index_col=None)
if choice == "Upload":
    st.title("Upload your data for modelling")
    file = st.file_uploader("Upload your dataset here!",type=["csv"])
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("source_data.csv",index=None)
        st.dataframe(df)
if choice == "Profiling":
    # #st.title("Exploratory Data Analysis")
    profile_report = ProfileReport(df,title="Exploratory data analysis")
    st_profile_report(profile_report)
if choice == "ML":
    st.title("Automating ML pipeline")
    target=st.selectbox("Select your target",df.columns)
    train, test = train_test_split(df,test_size=0.2,random_state=44)
    h2o.init()
    train_h2o = h2o.H2OFrame(train)
    test_h2o = h2o.H2OFrame(test)
    aml = H2OAutoML(max_models=20, seed=1234)
    aml.train(y=target, training_frame=train_h2o)
    lb = aml.leaderboard
    best_model = aml.get_best_model()
    pred = best_model.predict(test_h2o)
    perf = best_model.model_performance(test_h2o)

    st.title("H2O AutoML with Streamlit")
    st.write("Best Model:", best_model)
    st.write("Model Performance:", perf)
    st.write("Predictions:", pred)



if choice == "Download":
    pass
