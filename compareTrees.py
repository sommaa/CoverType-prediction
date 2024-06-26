from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

############################################
# compare different ML decision trees
############################################

# Load the dataset covtype
covtype = fetch_covtype()
X = covtype.data
y = covtype.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models
strModel = ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'BaggingClassifier']
models = [
    tree.DecisionTreeClassifier(max_depth=20),
    RandomForestClassifier(n_estimators=2, max_depth=16),
    ExtraTreesClassifier(n_estimators=2, max_depth=16),
    AdaBoostClassifier(n_estimators=50),
    BaggingClassifier(n_estimators=1)
]

############################################
# Train the models
############################################
# allocate vectors to store the performance metrics
accuracy = np.zeros(len(models))
# save the feature importance for each model in a dictionary
feature_importance = {}

for model in models:
    model.fit(X_train, y_train)

    # Evaluate the models
    y_pred = model.predict(X_test)
    # save the performance metrics for each model
    accuracy[models.index(model)] = accuracy_score(y_test, y_pred)
    # save model to disk with pickle in modeLM folder
    # create folder modeLM if it does not exist
    if not os.path.exists('modelMLComp'):
        os.makedirs('modelMLComp')

    filename = 'modelMLComp/' + strModel[models.index(model)] + '.sav'
    pickle.dump(model, open(filename, 'wb'))

    # save the feature importance for the model
    if strModel[models.index(model)] != 'BaggingClassifier':
        feature_importance[strModel[models.index(model)]] = model.feature_importances_
    else:
        feature_importance[strModel[models.index(model)]] = np.mean([
    tree.feature_importances_ for tree in model.estimators_], axis=0)
    
################################################
# Evaluate the models
################################################
# compare the performance of the models with an horizontal bar plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.barh(strModel, accuracy)
plt.xlabel('Accuracy')
plt.title('Comparison of different ML decision trees')
plt.show()

# plot the feature importance for each model as a subplot of horizontal matrices heatmap
import seaborn as sns
fig, axs = plt.subplots(5, 1, figsize=(15, 15))
fig.suptitle('Feature importance for each model')
for i in range(5):
    sns.heatmap(feature_importance[strModel[i]].reshape(1, -1)/np.max(feature_importance[strModel[i]]), ax=axs[i], cmap='viridis')
    axs[i].set_title(strModel[i])
    # disable the y-axis for all subplots
    axs[i].set_yticks([])
    if i == 4:
        axs[i].set_xlabel('Feature index')
    else:
        #disable the x-axis for all subplots except the last one
        axs[i].set_xticks([])
    # set the height of each subplot matrix to 1
    axs[i].set_aspect(1)
    # disable colorbar for all subplots
    axs[i].collections[0].colorbar.remove()
plt.show()

################################################################
# compare n_estimators vs max_depth for RandomForestClassifier
################################################################

# allocate vectors to store the performance metrics
accuracy = np.zeros((5, 7))

for i in range(1, 6):
    for j in range(1, 8):
        model = RandomForestClassifier(n_estimators=i**3, max_depth=j*5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy[i-1, j-1] = accuracy_score(y_test, y_pred)
        print('n_estimators =', i**3, 'max_depth =', j*5, 'accuracy =', accuracy[i-1, j-1])

# plot the matrix of accuracy
plt.figure(figsize=(12, 8))
# add annotations to the heatmap
max_depth = np.arange(5, 40, 5)
n_estimators = np.arange(1, 6)**3
sns.heatmap(accuracy, annot=True, cmap='Blues', annot_kws={"size": 12}, xticklabels=max_depth, yticklabels=n_estimators)
# plt.xlabel('max_depth = (i+1)*5')
plt.xlabel('max_depth', fontsize=20)
# plt.ylabel('n_estimators = (i+1)^3')
plt.ylabel('n_estimators', fontsize=20)
plt.show()

# save accuracy matrix to disk in modelMLComp folder with numpy
np.save('modelMLComp/accuracyMatrix.npy', accuracy)
