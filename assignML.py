from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# set the parameters
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rescale = True
# rebalance the dataset upsampling or downsampling
# 'none', 'SMOTE', 'SMOTEENN', 'SMOTETomek', 'RandomUnderSampler'
rebalance_method = 'none' 
save_model = True
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

############################################################################################
# preprocess and fit the models
############################################################################################

# Load the dataset covtype
covtype = fetch_covtype()
X = covtype.data
y = covtype.target

# no rescale or rebalance
# BAGGING CLASSIFIER
# avg on 20 runs without rescale (avg time = 5min on Intel i7-12700H)
# perfomance: mavg precision = 0.950, std = 0.004
#             wavg precision = 0.969, std = 0.002
#             mavg recall = 0.931, std = 0.001
#             wavg recall = 0.966, std = 0.002
#             mavg f1-score = 0.938, std = 0.003
#             wavg f1-score = 0.968, std = 0.001
#             accuracy = 0.966, std = 0.002

# DECISION TREE CLASSIFIER
# avg on 20 runs without rescale (avg time = 2min on Intel i7-12700H)
# perfomance: mavg precision = 0.889, std = 0.004
#             wavg precision = 0.922, std = 0.003
#             mavg recall = 0.864, std = 0.002
#             wavg recall = 0.915, std = 0.002
#             mavg f1-score = 0.865, std = 0.002
#             wavg f1-score = 0.918, std = 0.002
#             accuracy = 0.915, std = 0.002

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if rescale:
    # Normalize the dataset
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # BAGGING CLASSIFIER
    # avg on 20 runs with rescale (avg time = 5min on Intel i7-12700H)
    # perfomance: mavg precision = 0.951, std = 0.003
    #             wavg precision = 0.972, std = 0.002
    #             mavg recall = 0.929, std = 0.001
    #             wavg recall = 0.966, std = 0.001
    #             mavg f1-score = 0.940, std = 0.002
    #             wavg f1-score = 0.968, std = 0.001
    #             accuracy = 0.966, std = 0.001

    # DECISION TREE CLASSIFIER
    # avg on 20 runs with rescale (avg time = 2min on Intel i7-12700H)
    # perfomance: mavg precision = 0.891, std = 0.004
    #             wavg precision = 0.924, std = 0.003
    #             mavg recall = 0.863, std = 0.002
    #             wavg recall = 0.918, std = 0.002
    #             mavg f1-score = 0.866, std = 0.002
    #             wavg f1-score = 0.919, std = 0.002
    #             accuracy = 0.918, std = 0.002

# Apply the sampling methods only on the training set to avoid data contamination
if rebalance_method == 'SMOTE':
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    # BAGGING CLASSIFIER
    # avg on 10 runs with SMOTE (avg time = 23min on Intel i7-12700H)
    # perfomance: mavg precision = 0.922, std = 0.005
    #             wavg precision = 0.963, std = 0.003
    #             mavg recall = 0.937, std = 0.003
    #             wavg recall = 0.961, std = 0.003
    #             mavg f1-score = 0.925, std = 0.003
    #             wavg f1-score = 0.962, std = 0.002
    #             accuracy = 0.961, std = 0.002

    # DECISION TREE CLASSIFIER
    # avg on 10 runs with SMOTE (avg time = 8min on Intel i7-12700H)
    # perfomance: mavg precision = 0.781, std = 0.004
    #             wavg precision = 0.878, std = 0.002
    #             mavg recall = 0.911, std = 0.003
    #             wavg recall = 0.869, std = 0.004
    #             mavg f1-score = 0.830, std = 0.003
    #             wavg f1-score = 0.872, std = 0.002
    #             accuracy = 0.869, std = 0.002

elif rebalance_method == 'SMOTEENN':
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN()
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)
    # BAGGING CLASSIFIER
    # avg on 10 runs with SMOTEENN (avg time = 35min on Intel i7-12700H)
    # perfomance: mavg precision = 0.881, std = 0.005
    #             wavg precision = 0.944, std = 0.003
    #             mavg recall = 0.938, std = 0.003
    #             wavg recall = 0.943, std = 0.003
    #             mavg f1-score = 0.909, std = 0.003
    #             wavg f1-score = 0.939, std = 0.002
    #             accuracy = 0.943, std = 0.003

    # DECISION TREE CLASSIFIER
    # avg on 10 runs with SMOTEENN (avg time = 10min on Intel i7-12700H)
    # perfomance: mavg precision = 0.761, std = 0.004
    #             wavg precision = 0.866, std = 0.002
    #             mavg recall = 0.913, std = 0.003
    #             wavg recall = 0.858, std = 0.004
    #             mavg f1-score = 0.822, std = 0.003
    #             wavg f1-score = 0.870, std = 0.002
    #             accuracy = 0.858, std = 0.004


elif rebalance_method == 'SMOTETomek':
    from imblearn.combine import SMOTETomek
    smote_tomek = SMOTETomek()
    X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    # BAGGING CLASSIFIER
    # avg on 10 runs with SMOTETomek (avg time = 38min on Intel i7-12700H)
    # perfomance: mavg precision = 0.915, std = 0.005
    #             wavg precision = 0.959, std = 0.002
    #             mavg recall = 0.938, std = 0.003
    #             wavg recall = 0.959, std = 0.002
    #             mavg f1-score = 0.926, std = 0.004
    #             wavg f1-score = 0.959, std = 0.002
    #             accuracy = 0.959, std = 0.004

    # DECISION TREE CLASSIFIER
    # avg on 10 runs with SMOTETomek (avg time = 11min on Intel i7-12700H)
    # perfomance: mavg precision = 0.771, std = 0.004
    #             wavg precision = 0.888, std = 0.003
    #             mavg recall = 0.900, std = 0.003
    #             wavg recall = 0.869, std = 0.004
    #             mavg f1-score = 0.823, std = 0.005
    #             wavg f1-score = 0.871, std = 0.003
    #             accuracy = 0.869, std = 0.004
elif rebalance_method == 'RandomUnderSampler':
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_resample(X_train, y_train)
    # BAGGING CLASSIFIER
    # avg on 10 runs with RandomUnderSampler (avg time = 1min on Intel i7-12700H)
    # perfomance: mavg precision = 0.627, std = 0.018
    #             wavg precision = 0.785, std = 0.009
    #             mavg recall = 0.864, std = 0.007
    #             wavg recall = 0.748, std = 0.007
    #             mavg f1-score = 0.699, std = 0.012
    #             wavg f1-score = 0.757, std = 0.008
    #             accuracy = 0.748, std = 0.014

    # DECISION TREE CLASSIFIER
    # avg on 10 runs with RandomUnderSampler (avg time = 1min on Intel i7-12700H)
    # perfomance: mavg precision = 0.562, std = 0.019
    #             wavg precision = 0.721, std = 0.011
    #             mavg recall = 0.802, std = 0.008
    #             wavg recall = 0.680, std = 0.011
    #             mavg f1-score = 0.631, std = 0.016
    #             wavg f1-score = 0.689, std = 0.010
    #             accuracy = 0.680, std = 0.017

# Define the model
modelBc = BaggingClassifier(n_estimators=40)
modelDTC = tree.DecisionTreeClassifier(max_depth=30, min_impurity_decrease=0.00001, min_samples_split=2, min_samples_leaf=1)

# Train the model
modelBc.fit(X_train, y_train)
modelDTC.fit(X_train, y_train)

############################################################################################
# Evaluate the models
############################################################################################
# Evaluate the model BaggingClassifier
# make predictions
y_pred = modelBc.predict(X_test)
# calculate accuracy
print('-----------------------------------------------------------------')
print('accuracy for BaggingClassifier: ')
print('-----------------------------------------------------------------')
print(accuracy_score(y_test, y_pred))
# calculate classification report
target_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
print('-----------------------------------------------------------------')
print('Classification report for BaggingClassifier')
print('-----------------------------------------------------------------')
print(classification_report(y_test, y_pred, target_names=target_names, digits=5))
print('-----------------------------------------------------------------')
print('Confusion matrix for BaggingClassifier')
print('-----------------------------------------------------------------')
print(confusion_matrix(y_test, y_pred))
print('-----------------------------------------------------------------')

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# plot the confusion matrix with dark background
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, norm=matplotlib.colors.LogNorm(), cmap='Blues',annot_kws={"size": 12})
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)
plt.show()

# Evaluate the model DecisionTreeClassifier
y_pred = modelDTC.predict(X_test)
print('-----------------------------------------------------------------')
print('accuracy for DecisionTreeClassifier:')
print(accuracy_score(y_test, y_pred))
print('-----------------------------------------------------------------')
print('Classification report')
print('-----------------------------------------------------------------')
print(classification_report(y_test, y_pred, target_names=target_names, digits=5))
print('-----------------------------------------------------------------')
print(confusion_matrix(y_test, y_pred))
print('-----------------------------------------------------------------')

# plot the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, norm=matplotlib.colors.LogNorm(), cmap='Blues',annot_kws={"size": 12})
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Actual', fontsize=20)
plt.show()

# plot the feature importance for each model as a subplot of horizontal matrices heatmap
import numpy as np
strModel = ['BaggingClassifier', 'DecisionTreeClassifier']
feature_importance = {}
feature_importance['BaggingClassifier'] = np.mean([tree.feature_importances_ for tree in modelBc.estimators_], axis=0)
feature_importance['DecisionTreeClassifier'] = modelDTC.feature_importances_
import seaborn as sns
fig, axs = plt.subplots(2, 1, figsize=(10, 3))
fig.suptitle('Feature importance for each model')
for i in range(2):
    sns.heatmap(feature_importance[strModel[i]].reshape(1, -1)/np.max(feature_importance[strModel[i]]), ax=axs[i], cmap='viridis')
    axs[i].set_title(strModel[i])
    # disable the y-axis for all subplots
    axs[i].set_yticks([])
    if i == 1:
        axs[i].set_xlabel('Feature index')
    else:
        #disable the x-axis for all subplots except the last one
        axs[i].set_xticks([])
    # set the height of each subplot matrix to 1
    axs[i].set_aspect(1)
    # disable colorbar for all subplots
    axs[i].collections[0].colorbar.remove()
plt.show()

############################################################################################
# save model to disk with pickle in modeLM folder
############################################################################################
if save_model:
    import pickle
    import os

    # create folder modeML if it does not exist
    if not os.path.exists('modelML'):
        os.makedirs('modelML')

    filename = 'modelML/BaggingClassifier.sav'
    pickle.dump(modelBc, open(filename, 'wb'))
    filename = 'modelML/DecisionTreeClassifier.sav'
    pickle.dump(modelDTC, open(filename, 'wb'))

    # print number of parameters of the model
    print('-----------------------------------------------------------------')
    print('parameters for each model')
    print('-----------------------------------------------------------------')
    print('BaggingClassifier')
    print(modelBc.get_params())
    print('DecisionTreeClassifier')
    print(modelDTC.get_params())

    # Get the total number of parameters for all base estimators
    # in the Bagging Classifier
    n_nodes = np.sum([estimator.tree_.node_count for estimator in modelBc.estimators_])
    n_leaves = np.sum([estimator.tree_.n_leaves for estimator in modelBc.estimators_])
    n_parameters = np.sum([estimator.tree_.node_count * 2 + estimator.tree_.n_leaves for estimator in modelBc.estimators_])
    print("Total number of parameters in the Bagging Classifier:", n_parameters)
    # Get the total number of parameters in the Decision Tree Classifier
    num_parameters = modelDTC.get_n_leaves()
    print('Total number of parameters in the Decision Tree Classifier:', num_parameters)
