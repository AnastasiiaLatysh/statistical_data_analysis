import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from individual_task.task_3.task_3 import X_train, X_test, y_train, y_test

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
plot_pacf(y_pred, lags=50)
pyplot.savefig('pacf_nn')
plot_acf(y_pred, lags=50)
pyplot.savefig('akf_nn')
