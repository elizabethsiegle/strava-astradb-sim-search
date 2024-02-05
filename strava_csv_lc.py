import requests
from dotenv import dotenv_values
import pandas as pd


config = dotenv_values(".env")

activities_url = "https://www.strava.com/api/v3/athlete/activities"

header = {'Authorization': 'Bearer ' + config.get('STRAVA_TOKEN')}
params = {'per_page': 200, 'page': 1} #max 200 per page, can only do 1 page at a time
my_dataset = requests.get(activities_url, headers=header, params=params).json() #activities
page = 0
for x in range(1,3): #loop through 4 pages of strava activities
    page +=1 
    params = {'per_page': 200, 'page': page}
    my_dataset += requests.get(activities_url, headers=header, params=params).json()   

activities = pd.json_normalize(my_dataset)
cols = ['name', 'type', 'distance', 'moving_time', 'total_elevation_gain', 'start_date']
activities = activities[cols]
activities = activities[activities["start_date"].str.contains("2022") == False] #remove items from 2022, only include workouts from 2023
activities = activities[activities["type"] == "Run"] # only runs
activities = activities[activities["total_elevation_gain"] > 82.0] # over 82 total_elevation_gain

activities.to_csv('activities.csv', index=False)

# convert meters to miles
m_conv_factor = 1609
def convert_to_miles(num):
    return ((num/m_conv_factor)*1000)/1000.0

data_run_df = pd.read_csv('activities.csv')
data_run_df['distance'] = data_run_df['distance'].map(lambda x: convert_to_miles(x))
#convert moving time secs to mins, hours

# rewrite activities.csv with miles
data_run_df.to_csv('activities.csv', index=False)
