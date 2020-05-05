# 2.3. Здійснити аналіз пропущених значень.

import pandas as pd

# 1. Просте усереднення
from individual_task.const import PATH_TO_DATA_SET_WITH_MISSING_DATA

arrest_data = pd.read_csv(PATH_TO_DATA_SET_WITH_MISSING_DATA)
print(f"Дані перед використанням методу простого усереднення: \n: {arrest_data.head(6)}")
arrest_data.fillna(round(arrest_data['age'].mean()), inplace=True)
print(f"Дані після використанням методу простого усереднення: \n: {arrest_data.head(6)}")

print("\n\n\n")

# 2. Інтерполяція
arrest_data = pd.read_csv(PATH_TO_DATA_SET_WITH_MISSING_DATA)
print(f"Дані перед використанням інтерполяції: \n: {arrest_data.head(6)}")
arrest_data_without_missing_values = arrest_data.interpolate(method='linear', limit_direction='both', inplace=True)
print(f"Дані після використання інтерполяції: \n: {arrest_data.head(6)}")
