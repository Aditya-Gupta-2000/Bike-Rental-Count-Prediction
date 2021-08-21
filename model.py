import pandas as pd
import numpy as np
import seaborn as sns
import pickle

df =pd.read_csv('hour.csv')
X = df.drop(columns=['instant', 'dteday', 'hr', 'weekday', 'cnt','atemp', 'casual', 'registered', 'weathersit', 'workingday', 'holiday', 'weekday', 'yr'],
     axis=1)
Y = df['cnt']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestRegressor
RFModel_Total_Count= RandomForestRegressor(n_estimators=100).fit(X_train,Y_train)

pickle.dump(RFModel_Total_Count,open('model_Total_Count.pkl','wb'))

model_Total_Count = pickle.load(open('model_Total_Count.pkl','rb'))

print(RFModel_Total_Count.predict([[1,1,0.23,0.11,0.34]]))

