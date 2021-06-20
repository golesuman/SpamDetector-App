from typing import Text
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame
import streamlit as st

st.write("""
# Spam Message Classifier Web Application
## This app classifies whether the given message is spam or not
""")
def user_text_input():
    Text=st.text_input("Enter the Message")
    data={
       'Text':Text,
    }
    features=pd.DataFrame(data,index=[0])
    return features
df_features=user_text_input()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
df=pd.read_csv('spamtext.csv')
df.dropna(axis=0,inplace=True)
df['Class'].replace({'spam':1,'ham':0},inplace=True)
print(df['Class'])
x=df['Text']
y=df['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
v=CountVectorizer()
x_c=v.fit_transform(x_train)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_c,y_train)
email=[
    f'{df_features}',
]
email_count=v.transform(email)
prediction=model.predict(email_count)

st.subheader("The given message is:")
if prediction[0]==1:
    st.write("Spam")
else:
    st.write('Not Spam')

