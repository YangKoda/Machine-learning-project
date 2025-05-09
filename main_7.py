# J74

# Importing the necessary modules
import numpy as np
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Loading the dataset using pandas
df = pd.read_csv('ObesityDataSet.csv')
pd.set_option('display.max_columns', None)

# Exploratory Data Analysis
# print(df.head())
# print(df.describe())

# Getting the features of the dataset as a list
df_features = df.columns.tolist()
# print(df_features)

# # Checking whether the dataset needs cleaning or not
# for feature in df_features:
#     print(f'{feature} has {df[feature].value_counts()} values\n\n')  # Counting the values for each features
#     print(
#         f'{feature} has {df[feature].isnull().sum()} null values\n\n')  # Checking the number of na values from each
#     # features
#     print(f'{feature} has the datatype {df[feature].dtype}')  # Printing the datatypes of each features

#
# # Printing the histogram of features
# for feature in df_features:
#     if df[feature].dtypes == 'int64':
#         df[feature].hist(bins=50, figsize=(20, 20))

# print(df["NObeyesdad"].value_counts())
# pd.DataFrame(df["NObeyesdad"].value_counts()).plot(kind='bar', figsize=(20, 10))
# plt.savefig('Obesity_classes.jpg')

# # Printing all the entries which have obesity class Normal_Weight
# print(df[df['NObeyesdad'] == 'Normal_Weight'])

# Printing the features which are non-numeric features
non_numeric_features_1 = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

label_encoder = LabelEncoder()

for feature in df.columns:
    if df[feature].dtype == 'object':
        df[feature] = label_encoder.fit_transform(df[feature])

X = df.drop(columns='NObeyesdad')
y = df['NObeyesdad']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)

# Selection of relevant features
# Creating a kNN model
kNN_model = KNeighborsClassifier(n_neighbors=3)

# Initializing SequentialFeatureSelector for forward selection
forward_selector = SequentialFeatureSelector(
    estimator=kNN_model,
    direction="forward",  # Performing forward selection
    scoring="accuracy",  # Using accuracy as scoring metric
    cv=5,  # 5-fold cross validation
    n_features_to_select=14
)

forward_selector.fit(X, y)
selected_feature_indices = forward_selector.get_support(indices=True)
forward_selected_features = X.columns[selected_feature_indices]
print(forward_selected_features)

backward_selector = SequentialFeatureSelector(
    estimator=kNN_model,
    direction="backward",
    scoring="accuracy",
    cv=5,
    n_features_to_select=12
)

backward_selector.fit(X, y)
selected_feature_indices = backward_selector.get_support(indices=True)
backward_selected_features = X.columns[selected_feature_indices]
print(backward_selected_features)

selected_features = []
for i in forward_selected_features:
    if i in backward_selected_features:
        selected_features.append(i)

print(f'Selected features: {selected_features}')

# Updating the X with selected features
X = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)
param_grid_kNN = dict(n_neighbors=[i for i in range(5, 21)])
grid_kNN = GridSearchCV(KNeighborsClassifier(), param_grid_kNN, cv=14, scoring='accuracy')
grid_kNN.fit(X_train, y_train)

# View the complete results
print(grid_kNN.cv_results_)

# Examine the best model
print(grid_kNN.best_score_)
print(grid_kNN.best_params_)
print(grid_kNN.best_estimator_)

# So we can choose k=5, according to the grid search
kNN_model2 = KNeighborsClassifier(n_neighbors=5)
kNN_model2.fit(X_train, y_train)
y_pred_kNN = kNN_model2.predict(X_test)

classes_in_y = label_encoder.classes_

# Displaying the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_kNN)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes_in_y)
print(f'Accuracy with kNN: {accuracy_score(y_test, y_pred_kNN)}')
print(f'Confusion matrix is {conf_matrix}')
disp.plot()
plt.savefig('k5NN_confusion_matrix.jpg')

# Implementing grid search for svm
param_grid_SVM = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'rbf']}

grid_SVM = GridSearchCV(SVC(), param_grid_SVM, refit=True, verbose=3)
grid_SVM.fit(X_train, y_train)
print(f'Best parameters: {grid_SVM.best_params_}')  # Best parameters: {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
print(f'Best score: {grid_SVM.best_score_}')

# Implementing Support Vector Machine based on the parameters from the grid search
svm = SVC(kernel=grid_SVM.best_params_['kernel'], gamma=grid_SVM.best_params_['gamma'], C=grid_SVM.best_params_['C'])
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f'Accuracy with SVM: {svm_accuracy}')

# Displaying confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_kNN)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm, display_labels=classes_in_y)
print(f'Accuracy with kNN: {accuracy_score(y_test, y_pred_svm)}')
print(f'Confusion matrix is {conf_matrix_svm}')
disp_svm.plot()
plt.savefig('svm_confusion_matrix.jpg')

plt.show()
