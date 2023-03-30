import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import warnings # This used to show warning messages. The warning module is actually a subclass of Exception which is a built-in 
#class in Python.
warnings.filterwarnings('ignore')

def categorizing(dat):
    cat = dat.astype('category').cat.codes
    return cat


df1 = pd.read_csv("https://raw.githubusercontent.com/aadarsh1810/Project-on-2019-Election-Dataset/main/Deployment/Elections.csv")

column = ['GENERAL VOTES', 'POSTAL VOTES', 'TOTAL VOTES', 'OVER TOTAL ELECTORS  IN CONSTITUENCY', 'OVER TOTAL VOTES POLLED  IN CONSTITUENCY']
scaler = MinMaxScaler()
X = df1[column]
y = df1['WINNER']

X = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

Rforest_model = RandomForestClassifier(random_state=2,max_depth=10,n_estimators=50)
Rforest_model.fit(x_train,y_train)

def predictor(values):
    dict1 = {1:"Winner",0:"Loser"}

    random = [values]
    random_test = scaler.transform(random)

    prediction = dict1[Rforest_model.predict(random_test)[0]]
    return prediction
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://ak6.picdn.net/shutterstock/videos/23827366/thumb/7.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.title("Elections")
parties = list(df1['PARTY'].value_counts().index)
for i in parties:
    st.markdown(f'<h2>{i + ": "}</h2>', unsafe_allow_html = True)
    df2 = df1[df1['PARTY'] == i].reset_index(drop = True)
    result_df = df2[['NAME', 'CONSTITUENCY']]
    results = []
    for j in range(df2.shape[0]):
        result = predictor([df2[z][j] for z in column])
        results.append(result)
    result_df['WINNER/LOSER'] = results
    st.write(result_df)
