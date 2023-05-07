import numpy as np
import pandas as pd
import datetime

def calculate_age(born, event):
    return event.year - born.year - ((event.month, event.day) < (born.month, born.day))


icu_stays = pd.read_csv('mimic3_demo/ICUSTAYS.csv', header=0, parse_dates=['intime', 'outtime'])
patients = pd.read_csv('mimic3_demo/PATIENTS.csv', header=0, parse_dates=['dob', 'dod', 'dod_hosp', 'dod_ssn'])
admissions = pd.read_csv('mimic3_demo/ADMISSIONS.csv', header=0)
chart_events = pd.read_csv('mimic3_demo/CHARTEVENTS.csv', header=0,
                           dtype={'valueuom': str, 'resultstatus':str, 'stopped':str},
                           low_memory=False,
                           parse_dates=['charttime', 'storetime'])

chart_events = chart_events.dropna(subset=['icustay_id'])

#get earliest events per visit by sorting by icustay_id then event time, dropping all events for each stay except the last one, and renaming charttimes col
latest_events = chart_events[['subject_id', 'icustay_id', 'charttime']].sort_values(['charttime']).drop_duplicates(subset='icustay_id', keep='last').rename(columns={'charttime': 'latest_time'})
chart_events = chart_events.merge(latest_events[['icustay_id', 'latest_time']], left_on='icustay_id', right_on='icustay_id')
chart_events['timedelta_from_end'] = chart_events['latest_time'] - chart_events['charttime']

# test = chart_events[['subject_id', 'icustay_id', 'charttime']].sort_values(['charttime'])
# test2 = chart_events[['subject_id', 'icustay_id', 'charttime']].sort_values(['charttime']).drop_duplicates(subset='icustay_id', keep='last')
# print(test.head(30))
# print(test2.head(5))
# print(chart_events[['subject_id', 'charttime', 'latest_time', 'timedelta_from_end']].head(30))

time_stats = chart_events['timedelta_from_end'].describe()

#print(time_stats)


chart_events['hours_from_end'] = chart_events['timedelta_from_end'].dt.total_seconds() // 3600
#print(chart_events[['subject_id', 'charttime', 'timedelta_from_end', 'hours_from_end']].head(30))

chart_events = chart_events[chart_events['hours_from_end'] < 48].sort_values(['icustay_id', 'hours_from_end'])
print(chart_events.head(50))
print(len(chart_events))