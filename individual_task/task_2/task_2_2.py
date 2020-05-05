# 2.2. Визначити характеристики розподілів для числових даних (мода,
# медіана, математичне сподівання, дисперсія, асиметрія, ексцес, тощо).

import pandas as pd

from individual_task.const import PATH_TO_DATA_SET

arrest_data = pd.read_csv(PATH_TO_DATA_SET)

# 1. Mode (мода)
age_mode = arrest_data['age'].mode()[0]
year_mode = arrest_data['year'].mode()[0]
checks_mode = arrest_data['checks'].mode()[0]
print(f"Mode:\nage={age_mode}\nyear={year_mode}\nchecks={checks_mode}\n")

# 2. Median (медіана)
age_median = arrest_data['age'].median()
year_median = arrest_data['year'].median()
checks_median = arrest_data['checks'].median()
print(f"Median:\nage={age_median}\nyear={year_median}\nchecks={checks_median}\n")

# 3. Mean (математичне сподівання)
age_mean = arrest_data['age'].mean()
year_mean = arrest_data['year'].mean()
checks_mean = arrest_data['checks'].mean()
print(f"Mean:\nage={age_mean}\nyear={year_mean}\nchecks={checks_mean}\n")

# 4. Variance (дисперсія)
age_var = arrest_data['age'].var()
year_var = arrest_data['year'].var()
checks_var = arrest_data['checks'].var()
print(f"Variance:\nage={age_var}\nyear={year_var}\nchecks={checks_var}\n")

# 5. Standard deviation (середнє квадратичне відхилення)
age_std = arrest_data['age'].std()
year_std = arrest_data['year'].std()
checks_std = arrest_data['checks'].std()
print(f"Standard deviation:\nage={age_std}\nyear={year_std}\nchecks={checks_std}\n")

# 6. Skewness (коефіцієнт асиметрії)
age_skew = arrest_data['age'].skew()
year_skew = arrest_data['year'].skew()
checks_skew = arrest_data['checks'].skew()
print(f"Skewness:\nage={age_skew}\nyear={year_skew}\nchecks={checks_skew}\n")

# 7. Kurtosis (коефіцієнт ексцесу)
age_kurtosis = arrest_data['age'].kurtosis()
year_kurtosis = arrest_data['year'].kurtosis()
checks_kurtosis = arrest_data['checks'].kurtosis()
print(f"Kurtosis:\nage={age_kurtosis}\nyear={year_kurtosis}\nchecks={checks_kurtosis}\n")
