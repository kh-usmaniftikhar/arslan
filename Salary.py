from multimethod import distance
import streamlit as st
import pandas as pd
#from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from tangled_up_in_unicode import age

st.write("""
# Sallary Prediction App
This app predicts the Sallary based on age, distance and experience!
""")
#df = pd.read_csv("C:\Users\muyyassarhussain\Desktop\App_salary\ml_data_salary.csv")
#st.df.head(10)
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('age', 30.5, 35.5, 40.5)
    distance = st.sidebar.slider('distance', 70.0, 80.5, 100.4)
    YearsExperience = st.sidebar.slider('YearsExperience', 1.0, 6.9, 10.3)
    #petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 0.2)

    data = {
        'age' : age,
        'distance' : distance,
        'YearsExperience' : YearsExperience,
        #'petal_width' : petal_width
    }

    features = pd.DataFrame(data, index=[0])

    return features

df = user_input_features()

st.subheader('User Input Parameters')

st.write(df)

Salary =pd.read_csv('ml_data_salary.csv')

X = [[age,distance]]
Y = Salary

clf = RandomForestClassifier()
clf.fit(X, Y)

Prediction = clf.predict(df)

prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Salary.target_names)

st.subheader('Prediction')
st.write(Salary.target_names[Prediction])

st.subheader('Prediction Probablity')
st.write(prediction_proba)