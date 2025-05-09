# J74

# Importing the necessary modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from mrmr import mrmr_classif
from sklearn.metrics import mean_squared_error
from time import time

# Loading the dataset using pandas
df = pd.read_csv('ObesityDataSet.csv')
pd.set_option('display.max_columns', None)

start_time = time()

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


# pd.DataFrame(df["NObeyesdad"].value_counts()).plot(kind='bar', figsize=(20, 10))
# plt.savefig('Obesity_classes.jpg')

# # Printing all the entries which have obesity class Normal_Weight
# print(df[df['NObeyesdad'] == 'Normal_Weight'])

# Encoding the features...
# Encoding the dataset
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()

categorical_features = [i for i in df.columns if df[i].dtype == 'object' and i != 'NObeyesdad']
#
# for feature in df.columns:
#     if df[feature].dtype == 'object':
#         df[feature] = pd.get_dummies(df[feature])

if 'NObeyesdad' in categorical_features:
    categorical_features.remove('NObeyesdad')

X = one_hot_encoder.fit_transform(df[categorical_features])  # It becames a sparse matrix
y = label_encoder.fit_transform(df['NObeyesdad'])

# Mapping the encoded values to the classes
enc_classes = {}
for cl in label_encoder.classes_:
    for i in label_encoder.transform(label_encoder.classes_):
        enc_classes[i] = cl

# Converting the X back into Dataframe and concatenate with other features
X = pd.DataFrame(X.toarray(), columns=one_hot_encoder.get_feature_names_out(categorical_features))
X = pd.concat([df.drop(categorical_features + ['NObeyesdad'], axis=1), X], axis=1)

# Selection of relevant features using mRmR
selected_features = mrmr_classif(X, y, K=21)

print(f'Selected features: {selected_features}')

# Updating the X with selected features
X = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)

# Grid searching in cross validation for the optimal value for k
param_grid_kNN = dict(n_neighbors=[i for i in range(5, 21)])
grid_kNN = GridSearchCV(KNeighborsClassifier(), param_grid_kNN, scoring='accuracy')
grid_kNN.fit(X_train, y_train)

# View the complete results
print(grid_kNN.cv_results_)

# Examine the best model
print(grid_kNN.best_score_)
print(grid_kNN.best_params_)
print(grid_kNN.best_estimator_)

# # Creating a kNN model. So we can choose k=5, according to the grid search
kNN_model2 = KNeighborsClassifier(n_neighbors=5)
kNN_model2.fit(X_train, y_train)
y_pred_kNN = kNN_model2.predict(X_test)

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

# Plotting the ROC Curve: True Positive Rate vs False Positive Rate
# fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_svm, pos_label=2)
# plt.title('ROC SVM')
# plt.plot(fpr_svm, tpr_svm, label='ROC Curve SVM')
# plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate or (1 - Specifity)')
# plt.ylabel('True Positive Rate or (Sensitivity)')
# plt.title('Receiver Operating Characteristic')

# k vs MSE plot
ypredknn = {}
for i in range(1, 21):
    knn12 = KNeighborsClassifier(n_neighbors=i)
    knn12.fit(X_train, y_train)
    ypredknn[i] = knn12.predict(X_test)

mse_vs_k = {}

for i in range(1, 21):
    mse_vs_k[i] = mean_squared_error(y_test, ypredknn[i])

plt.plot(mse_vs_k.keys(), mse_vs_k.values())
plt.xlabel('k')
plt.ylabel('MSE')
plt.title('Mean Squared Error vs k-Value')
plt.xticks([i for i in range(1, 21)])
plt.savefig('mse_vs_k.png')

# Displaying the confusion matrix
conf_matrix_knn = confusion_matrix(y_test, y_pred_kNN, normalize='all')
conf_matrix_knn_display = ConfusionMatrixDisplay(conf_matrix_knn, display_labels=enc_classes.values())
# plt.title('Confusion Matrix - KNN, K=5 ')
conf_matrix_knn_display.plot()
plt.savefig('conf_matrix_knn.png')

conf_matrix_svm = confusion_matrix(y_test, y_pred_svm, normalize='all')
conf_matrix_svm_display = ConfusionMatrixDisplay(conf_matrix_svm, display_labels=enc_classes.values())
# plt.title('Confusion Matrix - SVM')
conf_matrix_svm_display.plot()
plt.savefig('conf_matrix_svm.png')

# Displaying the class value count distribution
# pd.DataFrame(y).value_counts().plot(kind='bar')
# pd.DataFrame(y_train).value_counts().plot(kind='bar')
# pd.DataFrame(y_test).value_counts().plot(kind='bar')

# Creating a pairplot
plt.figure(figsize=(10, 10))
sns.pairplot(df, hue='NObeyesdad')

# Creating the box plots of numerical features against target feature
num_features = [i for i in df.columns.tolist() if df[i].dtype != 'object']
plt.figure(figsize=(33, 33))
for feature in num_features:
    sns.boxplot(x='NObeyesdad', y=feature, data=df)
    plt.xticks(rotation=45)
    plt.show()

plt.show()
end_time = time()
print(f'Elapsed time is {end_time - start_time}.')
