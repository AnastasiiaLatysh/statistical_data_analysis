# 2.3. Здійснити аналіз пропущених значень.

import pandas as pd

# 1. Просте усереднення
arrest_data = pd.read_csv(
    '/home/alaty/PycharmProjects/statistical_data_analysis/Arrests_for_Drugs_Possession_with_missing_data.csv')
print(arrest_data.head(6))
arrest_data.fillna(round(arrest_data['age'].mean()), inplace=True)
print(arrest_data.head(6))


# 2. Інтерполяція
arrest_data = pd.read_csv(
    '/home/alaty/PycharmProjects/statistical_data_analysis/Arrests_for_Drugs_Possession_with_missing_data.csv')
print(arrest_data.head(6))
arrest_data_without_missing_values = arrest_data.interpolate(method='linear', limit_direction='both')
print(arrest_data_without_missing_values.head(6))
