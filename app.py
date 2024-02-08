import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import streamlit as st

st.set_page_config(
    page_title="Langauge Identification Model",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("House Price Prediction Model")

data =pd.read_csv("./housing.csv")

data.dropna(inplace=True)

X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']


#splitting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_data = X_train.join(y_train)

train_data.hist(figsize=(15, 8))

train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity, dtype=int)).drop(['ocean_proximity'], axis =1)
train_data

train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["households_rooms"] = train_data["total_rooms"] / train_data["households"]

X_train, y_train = train_data.drop(["median_house_value"], axis=1), train_data["median_house_value"]

reg = LinearRegression()

reg.fit(X_train, y_train)

test_data = X_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity, dtype=int)).drop(['ocean_proximity'], axis =1)

test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
test_data["households_rooms"] = test_data["total_rooms"] / test_data["households"]

# reg.predict([-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,"NEAR BAY"])


reg = LinearRegression()

reg.fit(X_train, y_train)

# test_data = X_test.join(y_test)

# Make predictions on the test set
# y_pred = reg.predict(test_data)

# accuracy = accuracy_score(y_test, y_pred)
# st.write(accuracy)