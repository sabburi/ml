# ml
Repo for code to run Georgia Tech Machine Learning(CS 7641) Assignment 1

Make sure the dataset is in the same folder as its accompanying code.

To run the various supervised learning algorithm on the apps dataset run:

$python3 apps.py

For the heart dataset, run:

$python3 heart.py

Both code takes in optional arguments:

DT
ANN
BOOST
KNN
SVM

Specify any number of these keywords as arguments to run those specific learning algorithms.
Running without any arguments will run all methods.

Example1:

$python3 heart.py DT ANN SVM

This will run only Decision Tree, Neural Nets, and SVM for the heart dataset.

Example2:

$python3 apps.py KNN

This will only run KNN for the apps dataset.

DT -> Decision Tree
ANN -> Multilayer Perceptron
BOOST -> Adaptive Boosting
KNN -> K Nearest Neighbors
SVM -> Support Vector Machine

