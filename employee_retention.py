# this challenge is incomplete. 
import pandas as pd
from datetime import datetime
import itertools
import numpy as np
import seaborn as sns

orig_data = pd.read_csv('../Challenges/employee_retention.csv')
orig_data['company_id'] = orig_data['company_id'].astype('category')
orig_data['employee_id'] = orig_data['employee_id'].astype('category')
# print(orig_data.isna().sum()) # initially there are 11192 records with NA quit date; no other columns have NAs
na_quit_date = pd.date_range(start='2015-12-14', end='2015-12-14', freq='D').to_list()[0]
orig_data['quit_date'] = orig_data.quit_date.fillna(na_quit_date)
print(orig_data.quit_date.isna().sum())

company_ids = orig_data.company_id.unique().tolist()
orig_data.describe() # describe doesn't show the description for categorical/string/date attributes
# generate a day by day headcount
# start with generating sequence of daily dates, for each company
start = datetime.strptime('2011-01-24', '%Y-%m-%d')
end = datetime.strptime('2015-12-13', '%Y-%m-%d')
date_lst = pd.date_range(start="2011-01-24",end="2015-12-13", freq='D').to_list()
date_comp_cross = pd.DataFrame(list(itertools.product(date_lst, company_ids)), columns=['date', 'company_id'])
date_comp_cross['date'] = date_comp_cross.apply(lambda x: str(x['date']).split(' ')[0], axis=1)

# create dataframe with number of people who joined on each date
num_ppl_joined = orig_data.groupby(['join_date']).size().to_frame('count').reset_index()
num_ppl_quit = orig_data.groupby(['quit_date']).size().to_frame('count').reset_index()
num_ppl_quit['quit_date'] = num_ppl_quit.apply(lambda x: str(x['quit_date']).split(' ')[0], axis=1)

date_comp_cross = pd.merge(date_comp_cross, num_ppl_joined, how='left', left_on='date', right_on='join_date')
date_comp_cross = pd.merge(date_comp_cross, num_ppl_quit, how='left', left_on='date', right_on='quit_date')
daily_join_quit = date_comp_cross.drop(columns=['join_date', 'quit_date']).rename(columns={'count_x': 'num_joined', 'count_y': 'num_quit'}).fillna(0)
daily_join_quit['num_employees'] = daily_join_quit['num_joined'] - daily_join_quit['num_quit']
daily_join_quit['num_employees'] = daily_join_quit.groupby(['company_id'])['num_employees'].cumsum()
# daily_emp['days_since_epoch'] = daily_emp.apply(lambda x: (datetime.strptime(x['date'], '%Y-%m-%d') - datetime(1970,1,1)).days, axis=1)