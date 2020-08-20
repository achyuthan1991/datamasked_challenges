import pandas as pd
import numpy as np

# read data
song_db = pd.read_json('../Challenges/song.json')
song_db.time_played = pd.to_datetime(song_db.time_played)
# # check size of data: 4000 rows/songs and 6 cols for each
# print(song_db.shape)
# # check if there are any nulls: No nulls
# print(song_db.isna().sum())
# # which are the states where users are based? how many states are there with users?
# print(song_db.user_state.unique())
# print(len(song_db.user_state.unique()))

# Question: Top 3 and bottom 3 states by number of users
users_by_state = song_db.groupby('user_state')['user_id'].agg(num_users=('user_id', pd.Series.nunique)).reset_index().sort_values(by='num_users')
num_states_with_users = len(song_db.user_state.unique())
# states with most users and least users:
print('States with highest users:')
print(users_by_state.iloc[num_states_with_users-3:num_states_with_users])   # states with most users
print('States with lowest users:')
print(users_by_state.iloc[0:3])   # states with least users


# find top 3 and bottom 3 states by User Engaegment
# define user engagement: number of times songs were played per user; can also be how many days did a user play songs
# at a state level: median/mean of number of songs played per user
song_db['date_played'] = song_db.time_played.dt.date
user_enagement = song_db.groupby(['user_id', 'user_state']).agg(num_times_played=('id', pd.Series.nunique), num_days_played=('date_played', pd.Series.nunique)).reset_index()
state_engaegment = user_enagement.groupby('user_state').agg(avg_times_played=('num_times_played', 'mean'), avg_days_played=('num_days_played', 'mean')).reset_index().sort_values(by=['avg_times_played', 'avg_days_played'])

# earliest signed up users from each state
# find earliest sign up date for each state; then find users who signed up on that date for each state
earliest_date_by_state = song_db.groupby('state').agg(earliest_date = ('user_sign_up_date', min)).reset_index()
earliest_user_songs = pd.merge(song_db, earliest_date_by_state, how='inner', left_on=['user_state', 'user_sign_up_date'], right_on=['user_state', 'earliest_date'])
earliest_users = earliest_user_songs[['user_state', 'user_id', 'user_sign_up_date']].drop_duplicates().sort_values(by='user_state')