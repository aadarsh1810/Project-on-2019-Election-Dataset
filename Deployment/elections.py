import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import warnings # This used to show warning messages. The warning module is actually a subclass of Exception which is a built-in 
#class in Python.
warnings.filterwarnings('ignore')

def categorizing(dat):
    cat = dat.astype('category').cat.codes
    return cat


df = pd.read_csv("https://raw.githubusercontent.com/aadarsh1810/Project-on-2019-Election-Dataset/main/Deployment/Elections.csv")
df2 = pd.read_csv("https://raw.githubusercontent.com/aadarsh1810/Project-on-2019-Election-Dataset/main/Source%20code%20and%20data%20files/Dataset_for_EDA.csv")
df4 = pd.read_csv("https://raw.githubusercontent.com/aadarsh1810/Project-on-2019-Election-Dataset/main/Source%20code%20and%20data%20files/Nota_dataset_for_EDA.csv")

column = ['GENERAL VOTES', 'POSTAL VOTES', 'TOTAL VOTES', 'OVER TOTAL ELECTORS  IN CONSTITUENCY', 'OVER TOTAL VOTES POLLED  IN CONSTITUENCY']
scaler = MinMaxScaler()
X = df[column]
y = df['WINNER']

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


st.title("Indian Elections Analysis")

parties = list(df['PARTY'].value_counts().index)

tab1, tab2 = st.tabs(['Results','Analysis'])

for i in parties:
    tab1.markdown(f'<h2>{i + ": "}</h2>', unsafe_allow_html = True)
    df3 = df[df['PARTY'] == i].reset_index(drop = True)
    result_df = df3[['NAME', 'CONSTITUENCY']]
    results = []
    for j in range(df3.shape[0]):
        result = predictor([df3[z][j] for z in column])
        results.append(result)
    result_df['WINNER/LOSER'] = results
    tab1.write(result_df)

vote_share_top5 = df2.groupby('PARTY')['TOTAL VOTES'].sum().nlargest(5).index.tolist()
def vote_share(row):
    if row['PARTY'] not in vote_share_top5:
        return 'Other'
    else:
        return row['PARTY']
df2['Party New'] = df2.apply(vote_share,axis =1)
counts = df2.groupby('Party New')['TOTAL VOTES'].sum()
labels = counts.index
values = counts.values
pie = go.Pie(labels=labels, values=values, marker=dict(line=dict(width=1)))
layout = go.Layout(title='Partywise Vote Share')
fig = go.Figure(data=[pie], layout=layout)
tab2.plotly_chart(fig)

st_con_vt=df[['STATE','CONSTITUENCY','TOTAL ELECTORS']]
fig = px.sunburst(st_con_vt, path=['STATE','CONSTITUENCY'], values='TOTAL ELECTORS',
                  color='TOTAL ELECTORS',
                  color_continuous_scale='viridis_r')
fig.update_layout(title_text='Sunburst Image of State and Constituency by Voters')
tab2.plotly_chart(fig)


def change_name(text):
  if text=='NOTA':
    return 1
  else:
    return 0
df4['NAME_MODIFIED'] = df4['NAME'].map(change_name)
df4['NAME_MODIFIED'].value_counts()
nota = df4[df4['NAME_MODIFIED']==1]
nota = (nota.groupby('STATE')['NAME'].value_counts()).sort_values(ascending=False)
no_of_nota = nota.reset_index(drop=True).values
state = nota.index.droplevel(1)
fig = plt.figure(figsize=(15,7))
# fig.set_axis_labels('Count', 'State')
sns.barplot(x=nota,y=state,orient='h',palette='rainbow')
plt.title("State-wise NOTA Entries")
tab2.pyplot(fig)

fig = plt.figure()
sns.heatmap(df.corr(),annot=True)
plt.title('Correlations of the dataset',size=15)
tab2.pyplot(fig)
footer="""<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by Aadarsh Nayyer and Abhinav Kumar</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
