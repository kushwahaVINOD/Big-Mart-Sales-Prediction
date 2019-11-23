import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier



def preprocessData(data):
	
	item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
	print(item_avg_weight)
	
	def impute_weight(cols):
	    Weight = cols[0]
	    Identifier = cols[1]
	    
	    if pd.isnull(Weight):
	        return item_avg_weight['Item_Weight'][item_avg_weight.index == Identifier]
	    else:
	        return Weight

	def impute_size_mode(cols):
	    Size = cols[0]
	    Type = cols[1]
	    if pd.isnull(Size):
	        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
	    else:
	        return Size

	def impute_visibility_mean(cols):
	    visibility = cols[0]
	    item = cols[1]
	    if visibility == 0:
	        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
	    else:
	        return visibility


	print ('Orignal #missing: %d'%sum(data['Item_Weight'].isnull()))
	data['Item_Weight'] = data[['Item_Weight','Item_Identifier']].apply(impute_weight,axis=1).astype(float)
	print ('Final #missing: %d'%sum(data['Item_Weight'].isnull()))


	print ('Orignal #missing: %d'%sum(data['Outlet_Size'].isnull()))
	data['Outlet_Size'] = data[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
	print ('Final #missing: %d'%sum(data['Outlet_Size'].isnull()))

	print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
	data['Item_Visibility'] = data[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)

	print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))


	print('Original Categories:')
	print(data['Item_Fat_Content'].value_counts())
	print('\nModified Categories:')
	data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular','low fat':'Low Fat'})
	print(data['Item_Fat_Content'].value_counts())

	le = LabelEncoder()
	data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
	var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
	for i in var_mod:
		data[i] = le.fit_transform(data[i])
	
	data = pd.get_dummies(data, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
	data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
    #Drop unnecessary columns:
	#data.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
	#data.drop(['source'],axis=1,inplace=True)

def linear_regression():

	train_df = pd.read_csv('./dataset/Train.csv')
	test_df = pd.read_csv('./dataset/Test.csv')

	train_df=preprocessData(train_df)
	test_df=preprocessData(test_df)

	mean_sales = train_df['Item_Outlet_Sales'].mean()

	# baseline_submission = pd.DataFrame({
	#     'Item_Identifier':test_df['Item_Identifier'],
	#     'Outlet_Identifier':test_df['Outlet_Identifier'],
	#     'Item_Outlet_Sales': mean_sales
	# },columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])



	lr = LinearRegression(normalize=True)

	X_train = train_df.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
	Y_train = train_df['Item_Outlet_Sales']
	X_test = test_df.drop(['Item_Identifier','Outlet_Identifier'],axis=1).copy()

	lr.fit(X_train, Y_train)


	lr_pred = lr.predict(X_test)


	lr_accuracy = round(lr.score(X_train,Y_train) * 100,2)
	lr_accuracy


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



linear_regression()