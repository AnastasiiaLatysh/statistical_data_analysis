import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from individual_task.task_3.task_3 import X_train, X_test, y_train, y_test

naive_bayes = GaussianNB()

# fit the model with data
naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
plot_pacf(y_pred, lags=50)
pyplot.savefig('pacf_bayes')
plot_acf(y_pred, lags=50)
pyplot.savefig('akf_bayes')
