import pandas as pd
import numpy as np
import graphlab as gl
from sklearn.cross_validation import train_test_split


path = '/Users/pauloarantes/Drive/galvanize/_capstone/Datasets - Capstone/Jus Brasil/lawyer_interactions20160616v2.csv'

# Reading and parsing data from specified path
df = pd.read_csv(path, sep=';', header=None, usecols=range(14), low_memory=False)

# Naming the columns on the dataframe
cols = ['lawyer_id',
'is_hot_case',
'hot_lead_spent',
'text_message_sent',
'phone_visualized',
'legal_case_id',
'title',
'expertise',
'location',
'created_at',
'referrer',
'hiring_intent',
'can_afford',
'text']

df.columns = cols

# After performing initial EDA on notebook, creating a dataframe only with relevant features for the final score
score_cols = ['is_hot_case', 'hot_lead_spent', 'text_message_sent', 'phone_visualized', 'hiring_intent', 'can_afford']
scores = df[score_cols]

# Creating final rating score based on the following scale:
# 1 -> No interaction from lawyer
# 2 -> Lawyer visualized phone number
# 3 -> Lawyer sent a message
# 4 -> Lawyer visualized phone number and sent a message
# 5 -> Lawyer spent one hot credit to access information before 2 hours hold
# period when the case is flagged as 'hot' (about 25% of cases)
scores['rating'] = 1
scores['rating'][scores['phone_visualized'] == 1] = 2
scores['rating'][scores['text_message_sent'] == 1] = 3
scores['rating'][(scores['text_message_sent'] == 1) & (scores['phone_visualized'] == 1)] = 4
scores['rating'][scores['hot_lead_spent'] == 1] = 5

# Dropping remaining columns:
scores = scores.drop(['is_hot_case', 'can_afford', 'hiring_intent'], axis=1)

# Creating dataframe with lawyer_id, legal_case_id, and rating from lawyer
ratings = pd.concat([df[['lawyer_id', 'legal_case_id']], scores['rating']], axis=1)
