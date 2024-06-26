from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

##############################################################################################
# make the smallest machine learning model possible to get weighted accuracy of 90% or more
##############################################################################################

# Load the dataset covtype
covtype = fetch_covtype()
X = covtype.data
y = covtype.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
modelDt = tree.DecisionTreeClassifier(max_depth=75, min_impurity_decrease=0.0000215, min_samples_split=10, min_samples_leaf=10)

###########################################################################################
# Train the model
###########################################################################################
modelDt.fit(X_train, y_train)

###########################################################################################
# Evaluate the model
###########################################################################################
# make predictions
y_pred = modelDt.predict(X_test)
# calculate accuracy
print('-----------------------------------------------------------------')
print('Accuracy score')
print('-----------------------------------------------------------------')
print(accuracy_score(y_test, y_pred))
print('-----------------------------------------------------------------')
# calculate classification report
target_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
print('-----------------------------------------------------------------')
print('Classification report')
print('-----------------------------------------------------------------')
print(classification_report(y_test, y_pred, target_names=target_names, digits=5))
print('-----------------------------------------------------------------')
print(confusion_matrix(y_test, y_pred))
print('-----------------------------------------------------------------')

# save model to disk with pickle in modelML folder
import os
import pickle

# create folder modelMLSmall if it does not exist
if not os.path.exists('modelMLSmall'):
    os.makedirs('modelMLSmall')

filename = 'modelMLSmall/ML.sav'
pickle.dump(modelDt, open(filename, 'wb'))

# print the number of parameters
print('-----------------------------------------------------------------')
print('Number of parameters:', modelDt.get_n_leaves())

# print the depth of the tree
print('Depth of the tree:', modelDt.get_depth())

# plot the confusion matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
# enlarge xlabel and ylabel and annotation  size of heatmap to 20
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, norm=matplotlib.colors.LogNorm(), cmap='Blues',annot_kws={"size": 12})
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)
plt.show()