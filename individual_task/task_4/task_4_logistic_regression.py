import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from individual_task.task_3.task_3 import X_train, X_test, y_train, y_test

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
