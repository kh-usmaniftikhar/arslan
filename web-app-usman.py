from multimethod import distance
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tangled_up_in_unicode import age_long
# web Containers
header = st.container()
data_sets = st.container()
features = st.container()
model_training= st.container()

with header:
    st.title('House App')
    st.text('In the project we have used salary dataset')
    
with data_sets:
    st.header ('House price')
    st.text('This app predicts the house based on rooms, bedrooms and housing_median_age!')

    # Import data set
    df = pd.read_csv(r'housing.csv')
    df = df.dropna()
    st.write(df.head(10))
    st.subheader('Different values in the data')
    st.bar_chart(df['housing_median_age'].value_counts())
    st.subheader('rooms')
    st.bar_chart(df['total_bedrooms'].value_counts(10))

    # Bar plot
    st.subheader('total_bedrooms')
    st.bar_chart(df['total_bedrooms'].head(10))
    
with features:
    st.header ("These are our app features")
    st.text('we will use maximum features. Its quite easy.')
    st.markdown('1.**housing_median_age.')
    st.markdown("2. **total_rooms.")
    st.markdown("3. **total_bedrooms.")
    
with model_training:
    st.header("Model Training")
    st.text('We will train and test the dataset.')

    # making columns
    input, display= st.columns(2)
    
    # Pahlay coloumn main ap k pas selction point ho.
    max_depth = input.slider("rooms?", min_value=10, max_value=100, value=20, step=5)
    
    # N-estimater
    n_estimators = input.selectbox("How many tree are there in RF?", options=[50,100,200,300,'No-limit'])
    # adding list of features
    input.write (df.columns)
    # Input Features
    input_features = input.text_input('Which feature we should used?')
    
    #Machine Learning Model
    
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    # yahana per ham aik condition lagatay hain.
    if n_estimators == 'No-limit':
        model = RandomForestRegressor(max_depth=max_depth)
    else:
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    # Define X and Y
    X = df[['housing_median_age']]
    y = df[['median_house_value']]
    
    # fit our model
    model.fit(X,y)
    pred = model.predict(y)
    
# Display errors
display.subheader("Mean absolute error of the model is : ")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean squared error of the model is : ")
display.write(mean_squared_error(y, pred))
display.subheader("R square score of the model is : ")
display.write(r2_score(y, pred))
