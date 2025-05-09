# J74
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
from time import time
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
import seaborn as sns

start = time()

# Read data from the file
dataset = pd.read_csv('ObesityDataSet.csv')
X = dataset.drop(columns=['NObeyesdad'], axis=1)
y = dataset['NObeyesdad']

# Check for missing values in each column
missing_values = dataset.isnull().sum()
print(missing_values)

# Displaying dataset
print(type(X))
df_X = pd.DataFrame(X)
df_y = pd.Series(y)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df_X)

# Exploratory Data Analysis (EDA)
column_names = dataset.columns[:-1]
plt.figure(figsize=(20, 15))
for i, col in enumerate(df_X.columns, 1):
    plt.subplot(5, 5, i)
    plt.scatter(df_X[col], df_y, alpha=0.5)
    plt.title(f'{column_names[i-1]} vs Obesity')
    plt.xlabel(column_names[i-1])
    plt.ylabel('Target')
plt.tight_layout()
plt.show()

# Dropping some features
columns_to_drop = [15]
X_dropped = np.delete(X, columns_to_drop, axis=1)
df_X = pd.DataFrame(X_dropped)
print(df_X)

# Encoding categorical data with LabelEncoder
le = LabelEncoder()
columns_to_encode = ['Gender', 'family_history_with_overweight',
                     'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC']
X = X.drop(['MTRANS', 'Height'], axis=1)   # Height removed... Due to moderate correlation
X[columns_to_encode] = X[columns_to_encode].apply(le.fit_transform)
y = le.fit_transform(y)
print(X)
print(y)

# Feature Scaling
sc = StandardScaler()
columns_to_scale = [num_col for num_col in dataset.columns if dataset[num_col].dtype != 'object']
X[columns_to_scale] = sc.fit_transform(dataset[columns_to_scale])

# Define the number of splits
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define hyperparameter grids
param_grid_knn = {
    'n_neighbors': list(range(1, 31)),
}

param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf']
}

param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

param_grid_random_forest = {
    'n_estimators': [10, 50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': randint(10, 200),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

# Initialize dictionaries to store models and results
best_models = {}
accuracy_scores = {}

# Split the data into folds and perform hyperparameter tuning
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Hyperparameter tuning for each model
    grid_kNN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn)
    grid_kNN.fit(X_train, y_train)
    best_models['kNN'] = grid_kNN.best_estimator_

    grid_svm = GridSearchCV(estimator=SVC(), param_grid=param_grid_svm)
    grid_svm.fit(X_train, y_train)
    best_models['SVC'] = grid_svm.best_estimator_

    grid_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid_decision_tree)
    grid_dt.fit(X_train, y_train)
    best_models['DecisionTree'] = grid_dt.best_estimator_

    random_search_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=100, cv=5)
    random_search_rf.fit(X_train, y_train)
    best_models['RandomForest'] = random_search_rf.best_estimator_

    # Initialize classifiers and fit on training data
    classifiers = {
        'kNN': KNeighborsClassifier(n_neighbors=grid_kNN.best_params_['n_neighbors']),
        'SVC': SVC(C=grid_svm.best_params_['C'],
                   gamma=grid_svm.best_params_['gamma'],
                   kernel=grid_svm.best_params_['kernel']),
        'GaussianNB': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(criterion=grid_dt.best_params_['criterion'],
                                               max_depth=grid_dt.best_params_['max_depth'],
                                               min_samples_split=grid_dt.best_params_['min_samples_split'],
                                               min_samples_leaf=grid_dt.best_params_['min_samples_leaf']),
        'RandomForest': RandomForestClassifier(n_estimators=random_search_rf.best_params_['n_estimators'],
                                               criterion=random_search_rf.best_params_['criterion'],
                                               max_depth=random_search_rf.best_params_['max_depth'],
                                               min_samples_split=random_search_rf.best_params_['min_samples_split'],
                                               min_samples_leaf=random_search_rf.best_params_['min_samples_leaf'],
                                               random_state=74)
    }

    for name, classifier in classifiers.items():

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        plt.title(f'Confusion Matrix : {name}')
        plt.xlabel('y_test')
        plt.ylabel('y_pred')
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.savefig(f'confusion_matrix_{name}.png')

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

        accuracy_scores.setdefault(name, []).append(accuracy)
        # print(f'Confusion Matrix for {name}:\n{cm}')
        print(f'Accuracy Score of {name}: {accuracy}')
        print(f'Precision Score of {name}: {precision}')
        print(f'Recall Score of {name}: {recall}')
        print(f'F1 Score of {name}: {f1_score}')

# Calculate and print average accuracies
for name, scores in accuracy_scores.items():
    average_accuracy = np.mean(scores)
    print(f'Average accuracy for {name} across {n_splits} folds: {average_accuracy:.2f}')

end = time()
print(f'Total time taken: {(end-start)/60:.2f} minutes')
