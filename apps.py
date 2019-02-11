import numpy as np
import pandas as pd
import sys
from pandas import Series,DataFrame
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

cat_col = ['Category', 'Genres', 'ContentRating', 'InstallLevel']

DT = False
KNN = False
BOOST = False
ANN = False
SVM = False

if(len(sys.argv) > 1):
	for i in range(1, len(sys.argv)):
		if(sys.argv[i] == 'DT'):
			DT = True
		elif(sys.argv[i] == 'KNN'):
			KNN = True
		elif(sys.argv[i] == 'BOOST'):
			BOOST = True
		elif(sys.argv[i] == 'ANN'):
			ANN = True
		elif(sys.argv[i] == 'SVM'):
			SVM = True
else:
	exitLoop = False
	while(not exitLoop):
		verify = input("Are you sure you want to run all algorithms?(y/n)\n")
		if(verify == "y"):
			DT = True
			KNN = True
			BOOST = True
			ANN = True
			SVM = True
			exitLoop = True
		elif(verify == "n"):
			exit(1)
		else:
			print("Unrecognized input")





# Function importing Dataset 
def importdata(): 

	app_data = pd.read_csv("apps.csv")
	
	app_data = app_data.drop(['Id', 'App', 'Installs', 'ColumnLength'], axis = 1)

	for attr in cat_col:
		le = preprocessing.LabelEncoder()
		le.fit(app_data[attr].unique())
		app_data[attr] = le.transform(app_data[attr])

	# Printing the dataset shape 
	print("Dataset Length: ", len(app_data)) 
	print("Dataset Shape: ", app_data.shape) 
	
	# Printing the dataset obseravtions 
	print ("Dataset: ",app_data.head()) 
	return app_data 

# Function to split the dataset 
def splitdataset(app_data): 

	# Seperating the target variable 
	X = app_data.drop(['InstallLevel'], axis = 1).values
	Y = app_data['InstallLevel'].values 

	# Spliting the dataset into train and test 
	train_x, test_x, train_y, test_y = train_test_split( 
	X, Y, test_size = 0.3, random_state = 100) 
	
	return X, Y, train_x, test_x, train_y, test_y 


# Building Phase
data = importdata() 
X, Y, train_x, test_x, train_y, test_y = splitdataset(data) 

x = []
y = []
y_test = []
y_train = []


############### Decision Tree ###############################
if(DT):
	print("-----Starting DT--------")

	model_tree_gini_best = tree.DecisionTreeClassifier()
	model_tree_entropy_best = tree.DecisionTreeClassifier(criterion="entropy")
	model_tree_gini_random = tree.DecisionTreeClassifier(splitter="random")
	model_tree_entropy_random = tree.DecisionTreeClassifier(criterion="entropy",splitter="random")

	model_tree_gini_best = model_tree_gini_best.fit(train_x, train_y)
	model_tree_entropy_best = model_tree_entropy_best.fit(train_x, train_y)
	model_tree_gini_random = model_tree_gini_random.fit(train_x, train_y)
	model_tree_entropy_random = model_tree_entropy_random.fit(train_x, train_y)

	print("Gini/Best Score")
	print(model_tree_gini_best.score(train_x, train_y))

	print("Entropy/Best Score")
	print(model_tree_entropy_best.score(train_x, train_y))

	print("Gini/RandomScore")
	print(model_tree_gini_random.score(train_x, train_y))

	print("Entropy/RandomScore")
	print(model_tree_entropy_random.score(train_x, train_y))

	y_train = []
	for i in range(1, 20):
		model_tree_depth = tree.DecisionTreeClassifier(max_depth=i)
		model_tree_depth = model_tree_depth.fit(train_x, train_y)
		y_train.append(model_tree_depth.score(train_x, train_y))

	print("FINISHED TRAINING")

	print("Gini/Best Score")
	print(model_tree_gini_best.score(test_x, test_y))

	print("Entropy/Best Score")
	print(model_tree_entropy_best.score(test_x, test_y))

	print("Gini/RandomScore")
	print(model_tree_gini_random.score(test_x, test_y))

	print("Entropy/RandomScore")
	print(model_tree_entropy_random.score(test_x, test_y))

	x = []
	y = []

	for i in range(1, 20):
		model_tree_depth = tree.DecisionTreeClassifier(max_depth=i)
		model_tree_depth = model_tree_depth.fit(train_x, train_y)
		x.append(i)
		y.append(model_tree_depth.score(test_x, test_y))

	plt.title("Decision Tree Max Depth vs. Accuracy")
	plt.xlabel("Max Depth")
	plt.ylabel("Accuracy Score")
	plt.plot(x, y, label="Test Data")
	plt.plot(x, y_train, label="Training Data")
	plt.legend(loc='lower right')
	plt.show()

	x = []
	y_test = []
	y_train = []


############### KNN ###############################
if(KNN):
	print("-----Starting KNN--------")


	for i in range(1, 20):
		neigh = KNeighborsClassifier(n_neighbors=i)
		x.append(i)
		neigh.fit(train_x, train_y)
		y_train.append(neigh.score(train_x, train_y))
		y_test.append(neigh.score(test_x, test_y))
	plt.title("K-Nearest Neighbors Parameter Analysis")
	plt.xlabel("k")
	plt.ylabel("Accuracy Score")
	plt.plot(x, y_test, label="Test Data")
	plt.plot(x, y_train, label="Training Data")
	plt.legend(loc='lower right')
	plt.show()


	neigh_kd_man = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='kd_tree', p=1)
	neigh_kd_euc = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='kd_tree', p=2)
	neigh_bt_man = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='ball_tree',p=1)
	neigh_bt_euc = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='ball_tree',p=2)
	neigh_bf_man = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='brute',p=1)
	neigh_bf_euc = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='brute',p=2)

	neigh_kd_man.fit(train_x, train_y)
	neigh_kd_euc.fit(train_x, train_y)
	neigh_bt_man.fit(train_x, train_y)
	neigh_bt_euc.fit(train_x, train_y)
	neigh_bf_man.fit(train_x, train_y)
	neigh_bf_euc.fit(train_x, train_y)


	neigh_kd_man_d = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='kd_tree', p=1)
	neigh_kd_euc_d = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='kd_tree', p=2)
	neigh_bt_man_d = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='ball_tree',p=1)
	neigh_bt_euc_d = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='ball_tree',p=2)
	neigh_bf_man_d = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='brute',p=1)
	neigh_bf_euc_d = KNeighborsClassifier(n_neighbors=8, weights='distance', algorithm='brute',p=2)

	neigh_kd_man_d.fit(train_x, train_y)
	neigh_kd_euc_d.fit(train_x, train_y)
	neigh_bt_man_d.fit(train_x, train_y)
	neigh_bt_euc_d.fit(train_x, train_y)
	neigh_bf_man_d.fit(train_x, train_y)
	neigh_bf_euc_d.fit(train_x, train_y)

	print("KD + Manhattan + Uniform")
	print(neigh_kd_man.score(train_x, train_y))
	print("KD + Euclidean + Uniform")
	print(neigh_kd_euc.score(train_x, train_y))
	print("Ball Tree + Manhattan + Uniform")
	print(neigh_bt_man.score(train_x, train_y))
	print("Ball Tree + Euclidean + Uniform")
	print(neigh_bt_euc.score(train_x, train_y))
	print("Brute + Manhattan + Uniform")
	print(neigh_bf_man.score(train_x, train_y))
	print("Brute + Euclidean + Uniform")
	print(neigh_bf_euc.score(train_x, train_y))

	print("KD + Manhattan + Distance")
	print(neigh_kd_man_d.score(train_x, train_y))
	print("KD + Euclidean + Distance")
	print(neigh_kd_euc_d.score(train_x, train_y))
	print("Ball Tree + Manhattan + Distance")
	print(neigh_bt_man_d.score(train_x, train_y))
	print("Ball Tree + Euclidean + Distance")
	print(neigh_bt_euc_d.score(train_x, train_y))
	print("Brute + Manhattan + Distance")
	print(neigh_bf_man_d.score(train_x, train_y))
	print("Brute + Euclidean + Distance")
	print(neigh_bf_euc_d.score(train_x, train_y))

	print("TRAINING DONE")

	print("KD + Manhattan + Uniform")
	print(neigh_kd_man.score(test_x, test_y))
	print("KD + Euclidean + Uniform")
	print(neigh_kd_euc.score(test_x, test_y))
	print("Ball Tree + Manhattan + Uniform")
	print(neigh_bt_man.score(test_x, test_y))
	print("Ball Tree + Euclidean + Uniform")
	print(neigh_bt_euc.score(test_x, test_y))
	print("Brute + Manhattan + Uniform")
	print(neigh_bf_man.score(test_x, test_y))
	print("Brute + Euclidean + Uniform")
	print(neigh_bf_euc.score(test_x, test_y))

	print("KD + Manhattan + Distance")
	print(neigh_kd_man_d.score(test_x, test_y))
	print("KD + Euclidean + Distance")
	print(neigh_kd_euc_d.score(test_x, test_y))
	print("Ball Tree + Manhattan + Distance")
	print(neigh_bt_man_d.score(test_x, test_y))
	print("Ball Tree + Euclidean + Distance")
	print(neigh_bt_euc_d.score(test_x, test_y))
	print("Brute + Manhattan + Distance")
	print(neigh_bf_man_d.score(test_x, test_y))
	print("Brute + Euclidean + Distance")
	print(neigh_bf_euc_d.score(test_x, test_y))

############### Boosting ###############################
if(BOOST):
	print("-----Starting Boosting--------")

	x = []
	y_train = []
	for i in range(1, 30):
		adaboost_depth = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=i))
		adaboost_depth = adaboost_depth.fit(train_x, train_y)
		x.append(i)
		y_train.append(adaboost_depth.score(train_x, train_y))

	y_test = []
	for i in range(1, 30):
		adaboost_depth = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=i))
		adaboost_depth = adaboost_depth.fit(train_x, train_y)
		y_test.append(adaboost_depth.score(test_x, test_y))


	plt.title("AdaBoost Max Depth vs. Accuracy")
	plt.xlabel("Max Depth")
	plt.ylabel("Accuracy Score")
	plt.plot(x, y_test, label="Test Data")
	plt.plot(x, y_train, label="Training Data")
	plt.legend(loc='lower right')
	plt.show()


	x_estimate = []
	y_train_estimate = []
	for i in range(1, 30):
		adaboost_depth = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=4), n_estimators=i)
		adaboost_depth = adaboost_depth.fit(train_x, train_y)
		x_estimate.append(i)
		y_train_estimate.append(adaboost_depth.score(train_x, train_y))

	y_test_estimate = []
	for i in range(1, 30):
		adaboost_depth = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=4), n_estimators=i)
		adaboost_depth = adaboost_depth.fit(train_x, train_y)
		y_test_estimate.append(adaboost_depth.score(test_x, test_y))

	plt.title("AdaBoost Number of Estimators vs. Accuracy")
	plt.xlabel("Estimators")
	plt.ylabel("Accuracy Score")
	plt.plot(x_estimate, y_test_estimate, label="Test Data")
	plt.plot(x_estimate, y_train_estimate, label="Training Data")
	plt.legend(loc='lower right')
	plt.show()

############### Neural Network ###############################
if(ANN):
	print("-----Starting ANN--------")

	x = []
	y_train = []
	for i in range(1, 20):
		x.append(i)
		layers = [i + 1]
		nn_max_neuron = MLPClassifier(hidden_layer_sizes=(layers))
		nn_max_neuron = nn_max_neuron.fit(train_x, train_y)
		y_train.append(nn_max_neuron.score(train_x, train_y))

	y_test = []
	for i in range(1, 20):
		layers = [i + 1]
		nn_max_neuron = MLPClassifier(hidden_layer_sizes=(layers))
		nn_max_neuron = nn_max_neuron.fit(train_x, train_y)
		y_test.append(nn_max_neuron.score(test_x, test_y))

	plt.title("Neural Networks Max Number of Neurons vs. Accuracy")
	plt.xlabel("Number of Neurons")
	plt.ylabel("Accuracy Score")
	plt.plot(x, y_train, label="Training Data")
	plt.plot(x, y_test, label="Test Data")
	plt.legend(loc='lower right')
	plt.show()


	x = []
	y_train = []
	for i in range(1, 30):
		x.append(i)
		layers = []
		for j in range(i):
			layers.append(5)

		nn_max_neuron = MLPClassifier(hidden_layer_sizes=(layers))
		nn_max_neuron = nn_max_neuron.fit(train_x, train_y)
		y_train.append(nn_max_neuron.score(train_x, train_y))

	y_test = []
	for i in range(1, 30):
		layers = []
		for j in range(i):
			layers.append(5)
		nn_max_neuron = MLPClassifier(hidden_layer_sizes=(layers))
		nn_max_neuron = nn_max_neuron.fit(train_x, train_y)
		y_test.append(nn_max_neuron.score(test_x, test_y))


	plt.title("Neural Networks Number of Hidden Layers of Neurons vs. Accuracy")
	plt.xlabel("Number of Hidden Layers")
	plt.ylabel("Accuracy Score")
	plt.plot(x, y_train, label="Training Data")
	plt.plot(x, y_test, label="Test Data")
	plt.legend(loc='lower right')
	plt.show()

	layers = []
	for j in range(6):
		layers.append(5)

	nn_max_neuron_lbfgs_iden = MLPClassifier(hidden_layer_sizes=(layers), solver='lbfgs', activation='identity').fit(train_x, train_y)
	nn_max_neuron_lbfgs_log = MLPClassifier(hidden_layer_sizes=(layers), solver='lbfgs', activation='logistic').fit(train_x, train_y)
	nn_max_neuron_lbfgs_tanh = MLPClassifier(hidden_layer_sizes=(layers), solver='lbfgs', activation='tanh').fit(train_x, train_y)
	nn_max_neuron_lbfgs_relu = MLPClassifier(hidden_layer_sizes=(layers), solver='lbfgs', activation='relu').fit(train_x, train_y)

	nn_max_neuron_sgd_iden = MLPClassifier(hidden_layer_sizes=(layers), solver='sgd', activation='identity').fit(train_x, train_y)
	nn_max_neuron_sgd_log = MLPClassifier(hidden_layer_sizes=(layers), solver='sgd', activation='logistic').fit(train_x, train_y)
	nn_max_neuron_sgd_tanh = MLPClassifier(hidden_layer_sizes=(layers), solver='sgd', activation='tanh').fit(train_x, train_y)
	nn_max_neuron_sgd_relu = MLPClassifier(hidden_layer_sizes=(layers), solver='sgd', activation='relu').fit(train_x, train_y)

	nn_max_neuron_adam_iden = MLPClassifier(hidden_layer_sizes=(layers), solver='adam', activation='identity').fit(train_x, train_y)
	nn_max_neuron_adam_log = MLPClassifier(hidden_layer_sizes=(layers), solver='adam', activation='logistic').fit(train_x, train_y)
	nn_max_neuron_adam_tanh = MLPClassifier(hidden_layer_sizes=(layers), solver='adam', activation='tanh').fit(train_x, train_y)
	nn_max_neuron_adam_relu = MLPClassifier(hidden_layer_sizes=(layers), solver='adam', activation='relu').fit(train_x, train_y)


	print(nn_max_neuron_lbfgs_iden.score(train_x, train_y))
	print(nn_max_neuron_lbfgs_log.score(train_x, train_y))
	print(nn_max_neuron_lbfgs_tanh.score(train_x, train_y))
	print(nn_max_neuron_lbfgs_relu.score(train_x, train_y))

	print(nn_max_neuron_sgd_iden.score(train_x, train_y))
	print(nn_max_neuron_sgd_log.score(train_x, train_y))
	print(nn_max_neuron_sgd_tanh.score(train_x, train_y))
	print(nn_max_neuron_sgd_relu.score(train_x, train_y))

	print(nn_max_neuron_adam_iden.score(train_x, train_y))
	print(nn_max_neuron_adam_log.score(train_x, train_y)) 
	print(nn_max_neuron_adam_tanh.score(train_x, train_y))
	print(nn_max_neuron_adam_relu.score(train_x, train_y))
	print("TRAINING DONE")
	print(nn_max_neuron_lbfgs_iden.score(test_x, test_y))
	print(nn_max_neuron_lbfgs_log.score(test_x, test_y))
	print(nn_max_neuron_lbfgs_tanh.score(test_x, test_y))
	print(nn_max_neuron_lbfgs_relu.score(test_x, test_y))

	print(nn_max_neuron_sgd_iden.score(test_x, test_y))
	print(nn_max_neuron_sgd_log.score(test_x, test_y))
	print(nn_max_neuron_sgd_tanh.score(test_x, test_y))
	print(nn_max_neuron_sgd_relu.score(test_x, test_y))

	print(nn_max_neuron_adam_iden.score(test_x, test_y))
	print(nn_max_neuron_adam_log.score(test_x, test_y)) 
	print(nn_max_neuron_adam_tanh.score(test_x, test_y))
	print(nn_max_neuron_adam_relu.score(test_x, test_y))

############### SVM ###############################
if(SVM):
	print("-----Starting SVM--------")

	svm_linear_classifier = SVC(kernel="linear").fit(train_x, train_y)
	# svm_poly_classifier = SVC(kernel="poly").fit(train_x, train_y)
	svm_rbf_classifier = SVC(kernel="rbf").fit(train_x, train_y)
	# svm_sigmoid_classifier = SVC(kernel="sigmoid").fit(train_x, train_y)

	print("Linear Training Score")
	print(svm_linear_classifier.score(train_x, train_y))
	# print("Polynomial Training Score")
	# print(svm_poly_classifier.score(train_x, train_y))
	print("RBF Training Score")
	print(svm_rbf_classifier.score(train_x, train_y))
	# print("Sigmoid Training Score")
	# print(svm_sigmoid_classifier.score(train_x, train_y))

	print("Linear Test Score")
	print(svm_linear_classifier.score(test_x, test_y))
	# print("Polynomial Test Score")
	# print(svm_poly_classifier.score(test_x, test_y))
	print("RBF Test Score")
	print(svm_rbf_classifier.score(test_x, test_y))
	# print("Sigmoid Test Score")
	# print(svm_sigmoid_classifier.score(test_x, test_y))




