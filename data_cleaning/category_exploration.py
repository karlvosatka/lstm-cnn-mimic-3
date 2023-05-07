import numpy as np
import pandas as pd
import datetime
#NOTE: needed to install openpyxl via pip install openpyxl to create excel file here

icu_stays = pd.read_csv('mimic3_demo/ICUSTAYS.csv', header=0, parse_dates=['intime', 'outtime'])
patients = pd.read_csv('mimic3_demo/PATIENTS.csv', header=0, parse_dates=['dob', 'dod', 'dod_hosp', 'dod_ssn'])
admissions = pd.read_csv('mimic3_demo/ADMISSIONS.csv', header=0)
chart_events = pd.read_csv('mimic3_demo/CHARTEVENTS.csv', header=0,
                           dtype={'valueuom': str, 'resultstatus':str, 'stopped':str},
                           low_memory=False,
                           parse_dates=['charttime', 'storetime'])
d_items = pd.read_csv('mimic3_demo/D_ITEMS.csv', header=0)

chart_events = chart_events.dropna(subset=['icustay_id'])
chart_events = chart_events.dropna(subset=['valuenum'])

chart_events = chart_events.merge(d_items[['itemid', 'label']], left_on='itemid', right_on='itemid').rename(columns={'label':'event_label'}).drop(columns=['cgid', 'storetime', 'warning', 'error', 'resultstatus', 'stopped'])

print(chart_events.columns.values)
unique_ce_labels = chart_events[['itemid', 'event_label', 'valuenum', 'valueuom']]
event_label_counts = unique_ce_labels['event_label'].value_counts()
unique_ce_labels['event_label_count'] = chart_events['event_label'].map(event_label_counts)
unique_ce_labels = unique_ce_labels.drop_duplicates(subset='event_label').sort_values('event_label_count', ascending=False)

unique_ce_labels.to_excel('unique_ce_labels.xlsx', index=False)

# print(unique_ce_labels.head(10))
# print(unique_ce_labels.shape[0])

#NOTE - categories were manually chosen in unique_ce_labels.xlsx file based on reference to the categories described in the paper

#add categories to chart_events
categories = pd.read_excel('unique_ce_labels.xlsx').dropna(subset='category')
#print(categories.head(10))
chart_events = chart_events.merge(categories[['event_label', 'category', 'label_change']], left_on='event_label', right_on='event_label')

# #find average value of each chart_event across the dataset
# category_avgs = chart_events.groupby('category').aggregate({'valuenum': 'mean'})
# #print(category_avgs.head())
# category_avgs.reset_index().to_excel('category_avgs.xlsx')
# category_avgs = chart_events.groupby(['category', 'event_label']).aggregate({'valuenum': 'mean', 'min', 'max', 'std'})
# category_avgs.to_json('category_avgs.json', orient='index')

# NOTE - labels were changed based on analysis of readouts from this excel file in order to ensure that "alarms" were treated as base reads
# and the labels were added to the original unique_ce_events.xlsx file for ease of reference
#
# In the dataset, some events labelled as "alarm" indicate that a measure was taken because it was in a dangerous range. In order to prevent
# error in subsequent normalization based on deviation from average values by label, these categories need to be relabelled as a generic event.
# 
# Example: ABP Alarm [Low] has avg value of 86, with std of 11. This is really an abnormal event that should be added to the ABP Systolic measure set,
# to reflect the nature of the measurement.
# 
# The category_avgs excel and json files were created to confirm the consistency of units of measure across occurrences of a event_label (they are consistent)
# and to also confirm that the average output of "alarm" type events was significantly higher than "non-alarm" events where appropriate

# 1. change event_labels of all alarm type events
chart_events.loc[chart_events['label_change'].notnull(), 'event_label'] = chart_events['label_change']

# 2. calculate average and std by label
chart_events['event_label_mean'] = chart_events.groupby('event_label')['valuenum'].transform('mean')
chart_events['event_label_std'] = chart_events.groupby('event_label')['valuenum'].transform('std')
chart_events['event_label_norm'] = (chart_events['valuenum'] - chart_events['event_label_mean']) / chart_events['event_label_std']

# now we can just distribute by category and treat them all as same units!
print(chart_events['category'].unique())

category_dict = {'Temperature': 1,
                 'Respiratory': 2,
                 'Heart Rate': 3,
                 'BP sys': 4,
                 'BP dias': 5,
                 'Capillary Refill Rate': 6,
                 'Glucose': 7,
                 'pH': 9,
                 'PAP sys': 10,
                 'PAP dias': 11,
                 'GCS': 12,
                 'Weight': 13,
                 'Height': 14}