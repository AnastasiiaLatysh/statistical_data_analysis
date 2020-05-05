# 3. Розбити навчальні дані на навчальну та перевірочну вибірку (крос-
# валідація). Рекомендується використовувати такі системи ІАД для
# виконання інтелектуального завдання, як SAS Enterprise Miner, SPSS, R, Python)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from individual_task.const import PATH_TO_DATA_SET

arrest_data = pd.read_csv(PATH_TO_DATA_SET)

# Now, we will be splitting the following data into labels and features. Labels are the data which we want to predict
# and features are the data which are used to predict labels. Here, we have used age as the label for
# predicting temperatures in y, data other than age is taken as features using the drop() function in X.
y = arrest_data.released
X = arrest_data.drop(['released', 'record_id', 'year'], axis=1)

# last step would be splitting the data into train and test data, we will do that using train_test_split() function.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# data encoding
encoder = LabelEncoder()
for feature in ['colour', 'sex', 'employed', 'citizen']:
    X_train[feature] = encoder.fit_transform(X_train[feature])
    X_test[feature] = encoder.fit_transform(X_test[feature])
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
