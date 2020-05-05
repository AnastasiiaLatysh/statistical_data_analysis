# 2.1. Побудувати графіки розподілів значень для числових даних.

import pandas as pd
import plotly.express as px

from individual_task.const import PATH_TO_DATA_SET

arrest_data = pd.read_csv(PATH_TO_DATA_SET)


# 1. Графік та гістограмма розподілу взаємозв'язку між віком людіні, що зберігає наркотичні засоби,
# та кількістю попередніх засуджень
amount_of_checks_per_age = arrest_data.groupby('age')[['checks']].sum().reset_index()
checks_per_age_graph = px.line(amount_of_checks_per_age, x='age', y='checks',
                               title='Dependency between age and amount of checks')
checks_per_age_histogram = px.histogram(arrest_data, x='age', y='checks',
                                        title='Dependency between age and amount of checks')
checks_per_age_graph.show()
checks_per_age_histogram.show()


# 2. Графік та гістограмма розподілу взаємозв'язку між ріком та кількістю засудженіх із минулою історією увязнень
amount_of_checks_per_year = arrest_data.groupby('year')[['checks']].sum().reset_index()
checks_per_year_graph = px.line(amount_of_checks_per_year, x='year', y='checks',
                                title='Dependency between year and amount of checks')
checks_per_year_histogram = px.histogram(arrest_data, x='year', y='checks',
                                         title='Dependency between year and amount of checks')
checks_per_year_graph.show()
checks_per_year_histogram.show()


# 3. Графік та гістограмма розподілу взаємозв'язку між ріком та середнім віком засудженого
average_age_per_year = arrest_data.groupby('year')[['age']].mean().reset_index()
average_age_per_year_graph = px.line(average_age_per_year, x='year', y='age',
                                     title='Dependency between year and average age')
average_age_per_year_graph.show()
