import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def split_dataset(X,Y):
	return train_test_split(X, Y, test_size = 0.2,random_state = 0)


def preprocessData(data):
	data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
	data['num_years'] = data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)
	data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace = True)
	data['Item_Visibility'].fillna(data['Item_Visibility'].mean(), inplace=True)
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


def mean_squared_loss(xdata, ydata, weights):
	m = len(xdata)
	hypothesis = np.dot(xdata, weights)
	# ydata=ydata.reshape(m,1)
	loss = hypothesis - ydata
	mse = np.sum(loss ** 2) / m
	return mse


def mean_squared_gradient(xdata, ydata, weights):
	# print (xdata)
	xTrans = xdata.transpose()
	hypothesis = np.dot(xdata, weights)
	m = len(ydata)
	loss = hypothesis - ydata
	gradient = (2 * np.dot(xTrans, loss)) / m
	return gradient


def train(xdata, ytrain,xtest):
	alpha = 0.001
	epoch = 50000

	weights = np.zeros([1, xdata.shape[1]])
	xdata = xdata.astype(np.float)
	xtrain = (xdata - xdata.mean()) / xdata.std()
	ytrain = (ytrain - ytrain.mean()) /ytrain.std()
	m = len(xtrain)
	weights = weights.transpose()
	ytrain = ytrain.reshape(m, 1)
	loss = []
	iters = []
	for i in range(0, epoch):
		cost = mean_squared_loss(xtrain, ytrain, weights)
		print("Iteration %d | Cost: %f" % (i, cost))
		iters.append(i)
		loss_to_append = mean_squared_loss(xtrain, ytrain, weights)
		loss.append(loss_to_append)
		gradient = mean_squared_gradient(xtrain, ytrain, weights)
		weights = weights - (alpha * gradient)
	y_pred = np.matmul(xtest, weights)
	return weights,y_pred


def linear_regression(X_train, Y_train, X_test):
	lr = LinearRegression(normalize=True)
	hist=lr.fit(X_train, Y_train)
	lr_pred = lr.predict(X_test)
	# error_avg = np.sum(np.absolute(np.subtract(lr_pred, Y_train))) / len(Y_train)
	# print(error_avg)
	return lr, lr_pred

def neural_network2(X_train,Y_train,X_test):

	min_max_scaler = preprocessing.MinMaxScaler()
	X_scale = min_max_scaler.fit_transform(X_train)
	X_test = min_max_scaler.fit_transform(X_test)
	#Y_scale = min_max_scaler.fit_transform(Y_train)
	min_Y = np.amin(Y_train)
	#print (min_Y)
	max_y = np.amax(Y_train)
	scale=max_y-min_Y
	Y_scale=(Y_train-min_Y)/scale

	#X_train, X_test1, Y_train, Y_test1 = train_test_split(X_scale, Y_scale, test_size=0.2)
	X_train, X_val, Y_train, Y_val = train_test_split(X_scale, Y_scale, test_size=0.1)
	print(X_train.shape, X_val.shape, Y_train.shape,  Y_val.shape)

	#print(X_train[0])
	#print (Y_train.shape)
	#print(Y_train[0][0])

	model = Sequential([
		Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
		Dense(128, activation='relu'),
        Dense(128, activation='relu'),
		Dense(1, activation='sigmoid'),
	])
	model.compile(optimizer='sgd',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])
	hist = model.fit(X_train, Y_train,
					 batch_size=32, epochs=100,
					 validation_data=(X_val, Y_val))
	predicted = model.predict(X_test)
	predicted_unscaled = (predicted * scale) + min_Y
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.savefig('NeuralNet.png')
	return model,predicted_unscaled.reshape(-1)

def neural_network(X_train,Y_train,X_test):

	min_max_scaler = preprocessing.MinMaxScaler()
	X_scale = min_max_scaler.fit_transform(X_train)
	X_test = min_max_scaler.fit_transform(X_test)
	#Y_scale = min_max_scaler.fit_transform(Y_train)
	min_Y = np.amin(Y_train)
	#print (min_Y)
	max_y = np.amax(Y_train)
	scale=max_y-min_Y
	Y_scale=(Y_train-min_Y)/scale

	#X_train, X_test1, Y_train, Y_test1 = train_test_split(X_scale, Y_scale, test_size=0.2)
	X_train, X_val, Y_train, Y_val = train_test_split(X_scale, Y_scale, test_size=0.1)
	#print(X_train.shape, X_val.shape, Y_train.shape,  Y_val.shape)

	#print(X_train[0])
	#print (Y_train.shape)
	#print(Y_train[0][0])

	model = Sequential([
		Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
		Dense(64, activation='relu'),
		Dense(1, activation='sigmoid'),
	])
	model.compile(optimizer='sgd',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])
	hist = model.fit(X_train, Y_train,
					 batch_size=32, epochs=100,
					 validation_data=(X_val, Y_val))
	predicted = model.predict(X_test)
	predicted_unscaled = (predicted * scale) + min_Y
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.savefig('NeuralNet.png')
	return model,predicted_unscaled.reshape(-1)

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
	X_train = X_train.to_numpy()
	Y_train = Y_train.to_numpy()
	X_train, X_test, Y_train, Y_test = split_dataset(X_train, Y_train)

	lr_model, prediction_lr = linear_regression(X_train, Y_train, X_test)
	X_train2 = np.asarray(X_train[0])
	Y_train2 = np.asarray(Y_train[0])

	pred_train = lr_model.predict(X_train)
	pred_train=pred_train.reshape(-1)
	#print(Y_train.shape)
	error = np.absolute(np.subtract(pred_train, Y_train))
	#print("error shape:", error.shape)
	#print("error:", error)
	c=0
	for i in range(1,len(error)):
		if error[i] > 100:
			c=c+1
			X_train2 = np.vstack((X_train2,X_train[i]))
			Y_train2 =np.vstack((Y_train2,Y_train[i]))
	ratio=c/len(error)
	# X_train2.append(np.asarray(X_train[i]))
	# Y_train2.append(pd.DataFrame(Y_train.loc[[i]]), ignore_index=True)
	accuracy_lr=find_accuracy(prediction_lr,Y_test)

	model,prediction_nn=neural_network(X_train2, Y_train2,X_test)
	accuracy_nn = find_accuracy(prediction_nn, Y_test)

	#model,prediction_nn_full = neural_network(X_train, Y_train, X_test)
	#accuracy_nn_full = find_accuracy(prediction_nn_full, Y_test)
	print("Linear Regressor accuracy:", accuracy_lr)
	print("Neural Network accuracy:",accuracy_nn)
	#print("accuracy neural_net_full_data:", accuracy_nn_full)
	prediction_ensemble = (ratio * prediction_nn) + ((1-ratio) * prediction_lr)
	ensemble = find_accuracy(prediction_ensemble, Y_test)
	print("Division Ratio",ratio)
	print("Accuracy after Ensembling:", ensemble)
main()