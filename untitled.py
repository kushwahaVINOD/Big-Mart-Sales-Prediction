import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv,math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def split_dataset(X,Y):
	return train_test_split(X, Y, test_size = 0.2,random_state = 0)


def preprocessData(data):
	data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
	data['num_years'] = data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)
	data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace = True)
	data['Outlet_Size'].fillna('Medium',inplace = True)
	col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
	data_temp = pd.get_dummies(data, columns = col, drop_first = True)
	feat_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'num_years','Item_Fat_Content_Regular', 'Item_Type_Breads', 'Item_Type_Breakfast','Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods','Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks','Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat','Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods','Item_Type_Soft Drinks', 'Item_Type_Starchy Foods','Outlet_Size_Medium', 'Outlet_Size_Small','Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3','Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2','Outlet_Type_Supermarket Type3']
	X = data_temp[feat_cols]
	Y = data_temp['Item_Outlet_Sales'] 
	return X,Y

def rmse(pred,Y_train):
	return math.sqrt(mean_squared_error(Y_train,lr_pred))

def find_accuracy(pred,y):

	return 100-(np.sum(np.absolute(np.subtract(pred,y)))/np.sum(y))*100

def linear_regression(X_train,Y_train,X_test):



	lr = LinearRegression(normalize=True)
	lr.fit(X_train, Y_train)


	lr_pred = lr.predict(X_test)

	return lr_pred


def neural_network(X_train,Y_train,X_test):

	pass


def bagged_decision_tree(X,Y):
	seed = 7
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cart = DecisionTreeClassifier()
	num_trees = 100
	model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
	results = model_selection.cross_val_score(model, X, Y, cv=kfold)
	print(results.mean())

def random_forest(X,Y):
	seed = 7
	num_trees = 100
	max_features = 3
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
	results = model_selection.cross_val_score(model, X, Y, cv=kfold)
	print(results.mean())

def main():
	
	train_df = pd.read_csv('./dataset/Train.csv')
	X_train,Y_train=preprocessData(train_df)
	X_train, X_test, Y_train, Y_test=split_dataset(X_train,Y_train)
	print(X_train.shape)
	prediction=linear_regression(X_train,Y_train,X_test)
	accuracy=find_accuracy(prediction,Y_test)	    
	print(accuracy)
	#bagged_decision_tree(X_train,Y_train)

main()