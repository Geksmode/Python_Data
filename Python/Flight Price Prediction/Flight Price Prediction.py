"""
Created on Tue Feb 27 13:40:01 2024

@author: Geksmode
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pickle

train_data = pd.read_excel("C:\\Users\\Geksmode\\Desktop\\data\\Data_Train.xlsx")
train_data.dropna(inplace=True)

def newd(x):
    if x=="New Delhi":
        return "Delhi"
    else :
        return x


#Split Date of journey in Journey day and Journey Month
train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'],format='%d/%m/%Y').dt.day
train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'],format='%d/%m/%Y').dt.month

train_data.drop('Date_of_Journey',inplace=True,axis=1)

train_data.head()

# print(train_data.dtypes)
#Extracting hours and minutes from time

train_data['Dep_hour'] = pd.to_datetime(train_data["Dep_Time"]).dt.hour
train_data['Dep_min'] = pd.to_datetime(train_data["Dep_Time"]).dt.minute
train_data.drop('Dep_Time',inplace = True, axis=1)

train_data['Arrival_hour'] = pd.to_datetime(train_data["Arrival_Time"]).dt.hour
train_data['Arrival_min'] = pd.to_datetime(train_data["Arrival_Time"]).dt.minute
train_data.drop('Arrival_Time',inplace = True, axis=1)
train_data.head()

#Dropping the Duration column and extracting important info from it
    #Same format for the duration of time (if 30m = > 0h 30m if 2h => 2h 0m)
duration = list(train_data['Duration'])

for i in range (len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i] :
            duration[i] = duration [i] + " 0m"
        else :
            duration[i] = '0h ' + duration[i]

duration_hour = []
duration_min = []

for i in duration:
    split = i.split()
    h = split[0]
    m = split[1]
    duration_hour.append(int(h[:-1]))
    duration_min.append(int(m[:-1]))

train_data['Duration_hours'] = duration_hour
train_data['Duration_min'] = duration_min
train_data.drop('Duration',axis=1,inplace=True)


# sns.catplot(x='Airline',y='Price',data=train_data.sort_values('Price',ascending=False),kind='boxen',aspect=3,height=6)
# sns.catplot(x= "Source", y="Price", data= train_data.sort_values('Price',ascending =False), kind="boxen", aspect= 3,height= 6)
# sns.catplot(x= "Destination", y = "Price",data=train_data.sort_values('Price',ascending=False),kind ="boxen",aspect = 3, height = 6)

#Dummy Airline
airline = train_data[['Airline']]
airline = pd.get_dummies(airline,drop_first=True)
#Dummy source
source = train_data[["Source"]]
source = pd.get_dummies(source,drop_first=True)
source.head()


#Dummy Destination
destination = train_data[["Destination"]]
destination = pd.get_dummies(destination,drop_first = True)
destination.head()

#dropping crap columns

train_data.drop(["Route","Additional_Info"],inplace=True,axis=1)

#check values in the total stops column


#convert label into numbers

train_data["Total_Stops"].replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)

#combine all 4 data frame
data_train = pd.concat([train_data,airline,source,destination],axis=1)
data_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)
data_train.head()

X = data_train.drop('Price',axis=1)

y = data_train["Price"]

reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X,y)

plt.figure(figsize = (12,8))
feat_importances = pd.Series(reg.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# Random search of parameters, using 5 fold cross validation, search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid,
                               scoring='neg_mean_squared_error', n_iter = 10, cv = 5, 
                               verbose=1, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)

rf_random.best_params_

# Flight Price Prediction
prediction = rf_random.predict(X_test)

plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)

# plt.figure(figsize = (8,8))
# plt.scatter(y_test, prediction, alpha = 0.5)
# plt.xlabel("y_test")
# plt.ylabel("y_pred")
# plt.show()

file = open('flight_rf.pkl', 'wb')
pickle.dump(rf_random, file)
