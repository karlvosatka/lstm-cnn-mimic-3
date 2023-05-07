import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import datetime
import pickle
import time
import multiprocessing as mp
from p_tqdm import p_imap

def create_event_list(group):
    events = []
    for j in range(48):
        events.append([])
        for k in range(13):
            events[j].append([])
            events[j][k] = group.loc[(group.hours_from_beginning==j) & (group.category_num==k)]['event_label_norm'].values.tolist()
    return events

if __name__ == '__main__':
    start_time = time.monotonic()

    with open('chart_events_df.pickle', 'rb') as f:
        chart_events = pickle.load(f)

    # labels = chart_events[['icustay_id', 'readmit_label']].drop_duplicates(subset='icustay_id').sort_values('icustay_id')

    # with open('readmit_labels.pickle', 'wb') as f:
    #     pickle.dump(labels['readmit_label'].to_list(), f)
    
    chart_events = chart_events.sort_values(['icustay_id', 'hours_from_beginning', 'category_num'])

    hour_is_48 = chart_events.loc[chart_events.hours_from_beginning == 47]['icustay_id'].unique()
    print(len(hour_is_48))

    print(len(chart_events['icustay_id'].unique()))

    assert len(hour_is_48) == (len(chart_events['icustay_id'].unique()))



    print('creating events list...')
    events = []
    iter = p_imap(create_event_list, [data for icu_stay, data in chart_events.groupby('icustay_id')], num_cpus=10) #10 cpus for Karl's computer
    
    for result in iter:
        events.append(result)

    print('events list done!')
    # max_len for MIMIC-III demo is 48
    # final output should have shape (len_pids, 48, 13, max_len) 
    print(events)

    print('pickling events...')
    with open('events.pickle', 'wb') as f:
        pickle.dump(events, f)

    print('done pickling events!')

    end_time = time.monotonic()
    print(datetime.timedelta(seconds=end_time - start_time))