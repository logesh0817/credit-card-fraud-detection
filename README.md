Credit Card Fraud Detection

A binary classification project that detects fraudulent credit card transactions and compares several machine learning models.

Dataset
File: creditcard.csv (about 150 MB, not included in this repo because of GitHub's 100 MB file limit)
Download: search for "Credit Card Fraud Detection" on Kaggle and place creditcard.csv in the project folder.
Features: Time, V1–V28 (anonymised PCA components), Amount
Target: Class (0 = legitimate, 1 = fraud). The data is highly imbalanced, so look at precision, recall and the confusion matrix rather than accuracy alone.
Workflow
Load the data and check for duplicate rows
Split into train and test sets (70/30, random_state=42)
Train and evaluate each model with a classification report and confusion matrix
Models compared
Logistic Regression
K-Nearest Neighbors
Support Vector Classifier (RBF kernel)
Decision Tree
Random Forest
Gradient Boosting
XGBoost
Results

Test set: 85,443 transactions, of which only 136 are fraud. Scores below are for the fraud class (Class = 1).

Model	Precision	Recall	F1
XGBoost	0.93	0.82	0.87
Decision Tree	0.84	0.83	0.83
K-Nearest Neighbors	0.86	0.76	0.80
Support Vector Classifier	0.84	0.72	0.78
Logistic Regression	0.87	0.62	0.73
Gradient Boosting	0.67	0.46	0.54

What I learned

Accuracy is 1.00 for every model because 99.9% of transactions are legitimate, so it says nothing about fraud detection. Precision, recall and F1 on the fraud class are what matter.
XGBoost performed best. It caught 111 of 136 frauds with only 9 false alarms (25 frauds missed).
Logistic Regression had decent precision but missed 51 of 136 frauds, and Gradient Boosting performed worst.
Next steps: handle the class imbalance (for example SMOTE or class weights) and tune the decision threshold to improve recall.

Note: In the notebook's Random Forest cell, the model rf is created but the Decision Tree dt is trained and evaluated, so that cell repeats the Decision Tree results. Fix it by using rf.fit(...) and rf.predict(...) to get real Random Forest scores.

Getting started
bash
pip install numpy pandas matplotlib scikit-learn xgboost jupyter
jupyter notebook classi.ipynb
Files
File	Description
classi.ipynb	Full notebook: preprocessing, training and evaluation
