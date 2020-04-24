from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

# train autoregression
from individual_task.task_3.task_3 import X_train, y_train, y_test

model = AR(X_train)
model_fitted = model.fit()
