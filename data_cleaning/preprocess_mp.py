import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
import pickle
import time
from tqdm import tqdm
import multiprocessing as mp


def process_icustay(last_hour_data):
    hours_from_beginning = last_hour_data.iloc[0]['hours_from_beginning']
    full_fake_data = pd.DataFrame()
    max_row_id = 330641192
    for i in range(int(hours_from_beginning) + 1, 48):
        fake_data = last_hour_data.copy().reset_index()
        fake_data['hours_from_beginning'] = i
        fake_data['hours_from_end'] = -1
        fake_data['row_id'] = range(max_row_id + 1, max_row_id + 1 + len(fake_data.index))
        full_fake_data = full_fake_data.append(fake_data)
        max_row_id += len(fake_data.index)
    return full_fake_data
    


if __name__ == '__main__':
    start_time = time.monotonic()

    mimic_path = 'mimic3_subset/'

    icu_stays = pd.read_csv(mimic_path + 'ICUSTAYS.csv', header=0, parse_dates=['intime', 'outtime'])
    patients = pd.read_csv(mimic_path + 'PATIENTS.csv', header=0, parse_dates=['dob', 'dod', 'dod_hosp', 'dod_ssn'])
    icd_9 = pd.read_csv(mimic_path + 'DIAGNOSES_ICD.csv', header=0)
    chart_events = pd.read_csv(mimic_path + 'CHARTEVENTS.csv', header=0,
                            usecols=['row_id', 'subject_id', 'hadm_id', 'icustay_id', 'itemid', 'charttime', 'valuenum'],
                            parse_dates=['charttime']) #with this config, 56 bytes per row
    #d_items = pd.read_csv(mimic_path + 'D_ITEMS.csv', header=0)

    category_dict = {'Temperature': 0,
                    'Respiratory': 1,
                    'Heart Rate': 2,
                    'BP sys': 3,
                    'BP dias': 4,
                    'Capillary Refill Rate': 5,
                    'Glucose': 6,
                    'pH': 7,
                    'PAP sys': 8,
                    'PAP dias': 9,
                    'GCS': 10,
                    'Weight': 11,
                    'Height': 12}

    # To create following dictionaries: Add categories information from excel file (see category_exploration.py) and remove any non-categorized events
    # categories = pd.read_excel('unique_ce_labels.xlsx').dropna(subset='category')
    # chart_events = chart_events.merge(categories[['event_label', 'category', 'label_change']], left_on='event_label', right_on='event_label').dropna(subset='category')

    # Create label change dict with (itemid, new_item_id) referencing categories
    # hardcoded with reference to the following, where categories is imported from category_exploration.py output file unique_ce_labels.xlsx
    # change_reqd = categories.loc[~categories['label_change'].isna()][['event_label', 'label_change', 'itemid']]
    # print(change_reqd.head(20))

    label_change = {
        8549:220045, #HR Alarm [High] -> Heart Rate
        5815:220045, #HR Alarm [Low] -> Heart Rate
        220047:220045, #Heart Rate Alarm - Low -> Heart Rate
        220046:220045, #Heart rate Alarm - High -> Heart Rate

        220050:51, #Arterial Blood Pressure systolic -> Arterial BP [Systolic]
        5813:51, #ABP Alarm [Low] -> Arterial BP [Systolic]
        8547:51, #ABP Alarm [High] -> Arterial BP [Systolic]
        220056:51, #Arterial Blood Pressure Alarm - Low -> Arterial BP [Systolic]
        220058:51, #Arterial Blood Pressure Alarm - High -> Arterial BP [Systolic]
        225309:51, #ART BP Systolic -> Arterial BP [Systolic]
        6701:51, #Arterial BP #2 [Systolic] -> Arterial BP [Systolic]
        227538:51, #ART Blood Pressure Alarm - Low -> Arterial BP [Systolic]
        
        220051:8368, #Arterial Blood Pressure diastolic -> Arterial BP [Diastolic]
        225310:8368, #ART BP Diastolic -> Arterial BP [Diastolic]
        8555:8368, #Arterial BP #2 [Diastolic] -> Arterial BP [Diastolic]
    }

    # # Create dict from itemid to category
    # categories['category_num'] = categories['category'].map(category_dict)
    # item_id_to_category = categories.set_index('itemid')['category_num'].to_dict()
    # print(item_id_to_category)

    itemid_to_category_num = {
        220045: 2, 220210: 1, 220277: 5, 646: 5, 220179: 3, 
        220180: 4, 51: 3, 8368: 4, 8549: 2, 5815: 2, 5820: 5, 
        8554: 5, 5819: 1, 8553: 1, 8551: 3, 5817: 3, 581: 11, 
        455: 3, 8441: 4, 220050: 3, 220051: 4, 113: 2, 223753: 10, 
        5813: 3, 8547: 3, 220739: 10, 223900: 10, 223901: 10, 
        223761: 0, 184: 10, 723: 10, 454: 10, 198: 10, 5814: 2, 
        8548: 2, 220074: 2, 677: 0, 678: 0, 8448: 9, 492: 8, 807: 6, 
        676: 0, 679: 0, 615: 1, 219: 1, 220047: 2, 220046: 2, 223770: 5,
        223769: 5, 224161: 1, 224162: 1, 226253: 5, 225664: 6, 
        811: 6, 619: 1, 220293: 1, 220292: 1, 8552: 8, 5818: 8, 
        614: 1, 224689: 1, 1529: 6, 223751: 3, 223752: 3, 780: 7, 
        1126: 7, 224688: 1, 220056: 3, 220058: 3, 225309: 3, 225310: 4, 
        224639: 11, 226531: 11, 762: 11, 226512: 11, 220734: 7, 
        226537: 6, 226707: 12, 226730: 12, 223762: 0, 6701: 3, 
        8555: 4, 220060: 9, 220059: 8, 227538: 3, 220066: 8, 220063: 8}

    ################# I. Drop Empty Rows and Consolidate Data from Other Tables

    #1. Drop chartevents entry where we don't have a number in value_num or an icustay_id to reference
    chart_events = chart_events.dropna(subset=['valuenum'])
    chart_events = chart_events.dropna(subset=['icustay_id'])

    #2. Drop chartevents for which we don't have icu_stays data
    chart_events = chart_events[chart_events['subject_id'].isin(set(icu_stays['subject_id']))]

    #3. Replace any labels referenced in label_change

    # print(chart_events.loc[chart_events.itemid == 8549].head(10))
    # print(chart_events.loc[chart_events.itemid == 5820].head(1)) #should NOT be empty
    # print(chart_events.loc[chart_events.row_id == 235308484]) #item should be 8549
    chart_events['itemid'] = chart_events['itemid'].map(label_change).fillna(chart_events['itemid'])
    # print(chart_events.loc[chart_events.itemid == 8549].head(10)) #should be empty
    # print(chart_events.loc[chart_events.itemid == 5820].head(1)) #should NOT be empty, should have 5820
    # print(chart_events.loc[chart_events.row_id == 235308484]) #itemid should be 220045


    #4. Add category_num for later category sorting
    chart_events['category_num'] = chart_events['itemid'].map(itemid_to_category_num)
    # print(chart_events.loc[chart_events.itemid == 51]['category_num']) #should all be category_num of 3

    ################ II. Clean Data of Excluded Patients and Label Readmission Decision

    ### A. Finding those below age 18 

    #1. Find earliest chart event for every patient in chartevents
    unique_chart_event_patients = chart_events.sort_values(by=['charttime']).drop_duplicates(subset='subject_id', keep='first')
    #2. Add patients information (crucially, dob) to the less_than_eighteen df
    less_than_eighteen = unique_chart_event_patients.merge(patients, left_on='subject_id', right_on='subject_id')
    #3. Calculate age col using calculate_age function
    #less_than_eighteen['age'] = less_than_eighteen.apply(lambda row: calculate_age(row['dob'], row['charttime']), axis=1) #event.year - born.year - ((event.month, event.day) < (born.month, born.day))
    # below prevents int64 overflow, confirmed to same result as apply example above, just a quicker calculation
    less_than_eighteen['age'] = ((less_than_eighteen['charttime'].values  - less_than_eighteen['dob'].values).astype(np.int64)/8.64e13//365).astype(np.int64) 
    #print(less_than_eighteen['age'].head(20))
    #4. Create series with list of pids for patients less than 18 years old
    less_than_eighteen = less_than_eighteen[less_than_eighteen.age < 18]['subject_id'].to_list()


    ### B. Find those who die in ICU and those who die w/in 30 days

    #1. Merge icu_stays with patients and calculate time_diff
    thirty_day_dead_pos = icu_stays.merge(patients, left_on='subject_id', right_on='subject_id')
    thirty_day_dead_pos['time_diff'] = thirty_day_dead_pos['dod'] - thirty_day_dead_pos["outtime"]

    #2. First find all that died in icu for later removal. be sure to also remove from thirty_day_dead_pos before next step
    died_in_icu = thirty_day_dead_pos[thirty_day_dead_pos.time_diff <= datetime.timedelta(0)]['subject_id'].to_list()
    thirty_day_dead_pos = thirty_day_dead_pos[thirty_day_dead_pos.time_diff > datetime.timedelta(0)]

    #3. Then find all who died within 30 days of discharge
    thirty_day_dead_pos = thirty_day_dead_pos[thirty_day_dead_pos.time_diff <= datetime.timedelta(30)]['icustay_id'].to_list()

    ### C. Find patients with multiple stays w/in 30 days

    #1. Create df with only duplicate visits for the same subject_id
    multi_icu = icu_stays[icu_stays.duplicated(subset='subject_id', keep=False)].reset_index()

    #2. Sort by outtime to ensure when we pull from groupby that visits are sequential. don't need pid sort, groupby will handle
    multi_icu = multi_icu.sort_values(['outtime'])

    pos_returned_in_thirty = [] #list of icustay_ids

    #3. Find the time between leaving and returning the icu for the stays for each patient with mult stays
    for pid, group in multi_icu.groupby('subject_id'):
        intimes = pd.to_datetime(group['intime']).dt.date.unique().tolist()
        outtimes = pd.to_datetime(group['outtime']).dt.date.unique().tolist()
        icu_stay_ids = group['icustay_id'].tolist()[:-1] #if one stay is 30 hours before another stay, the earlier stay required readmission and is positive
        assert len(icu_stay_ids) == len(intimes[1:])
        diff = np.subtract(intimes[1:], outtimes[:-1])
        if any(x < datetime.timedelta(30) for x in diff):
            for index, x in enumerate(diff):
                if x < datetime.timedelta(30): pos_returned_in_thirty.append(icu_stay_ids[index])

    ### D. Remove as appropriate and add readmission label
    remove = died_in_icu + less_than_eighteen
    pos = pos_returned_in_thirty + thirty_day_dead_pos

    chart_events = chart_events[~chart_events['subject_id'].isin(remove)]
    chart_events['readmit_label'] = chart_events['icustay_id'].isin(pos).astype(int)
    print("after removing dead or young: ", len(chart_events['icustay_id'].unique()))

    ############### III. Normalize event value as standard deviation from mean on a per-event_label basis

    # 1. Calculate average and std by itemid
    chart_events['event_label_mean'] = chart_events.groupby('itemid')['valuenum'].transform('mean')
    chart_events['event_label_std'] = chart_events.groupby('itemid')['valuenum'].transform('std')

    # 2. norm to zero mean and scaled to std. Drop unnecessary columns
    chart_events['event_label_norm'] = (chart_events['valuenum'] - chart_events['event_label_mean']) / chart_events['event_label_std']
    chart_events = chart_events.drop(columns=['event_label_mean', 'event_label_std'])

    ################ IV. Chunk by hours before the last chart event

    # 1. Get the last event for each patient and add as column to chart_events
    latest_events = chart_events[['subject_id', 'icustay_id', 'charttime']].sort_values(['charttime']).drop_duplicates(subset='icustay_id', keep='last').rename(columns={'charttime': 'latest_time'})
    chart_events = chart_events.merge(latest_events[['icustay_id', 'latest_time']], left_on='icustay_id', right_on='icustay_id')

    # 2. Get the number of hours from the end of the icustay per patient
    chart_events['hours_from_end'] = (chart_events['latest_time'] - chart_events['charttime']).dt.total_seconds() // 3600

    # 3. Drop all events outside of 48 hours
    chart_events = chart_events[chart_events['hours_from_end'] < 48].sort_values(['icustay_id', 'hours_from_end', 'category_num'])


    ################ V. Pad patients with less than 48 hours of data

    ### A. Find patients with less than 48 hours of data

    # 1. Add hours_from_beginning to this new subset of events ocurring in the last 48 hours. For some events, not all hours will be represented,
    # but these hours are about to be padded.
    chart_events['max_hours_from_end'] = chart_events.groupby('icustay_id')['hours_from_end'].transform('max')
    chart_events['hours_from_beginning'] = chart_events['max_hours_from_end'] - chart_events['hours_from_end']
    chart_events = chart_events.drop(columns=['max_hours_from_end'])

    # 2. Get patients with at least one occurrence of 47
    full_time_stays = chart_events.loc[chart_events['hours_from_end'] == 47]['icustay_id'].unique()

    # 3. Get patients who aren't in the above list and find their max hours from beginning
    stays_to_expand = chart_events[~chart_events['icustay_id'].isin(full_time_stays)]
    print(len(stays_to_expand['icustay_id'].unique()))
    #the following line takes only the events that occur when hours from end is 0, aka from the last hour, and then groups them by icustay_id
    last_stays_to_expand = stays_to_expand.loc[stays_to_expand.hours_from_end == 0].groupby('icustay_id')
    print(last_stays_to_expand.ngroups)
    print('adding duplicated hour...')
    print("length of chart_events before adding duplicates: ", len(chart_events))
    # 4. Add data on patient by patient basis
    max_row_id = chart_events['row_id'].max()
    #print(max_row_id)
    
    pool = mp.Pool(10) #for Karl's computer
    
    results = []
    for result in tqdm(pool.imap_unordered(process_icustay, (data for icustay_id, data in last_stays_to_expand), chunksize=10), total=len(last_stays_to_expand)):
        chart_events = pd.concat([chart_events, result], ignore_index=True)
    pool.close()
    pool.join()
    

    print('done adding duplicated hour!')
    print("this is the number of stays with something at 47 hours from beginning. it should be 7331!", len(chart_events.loc[chart_events.hours_from_beginning == 47]['icustay_id'].unique()))
    print("this is the length of chart_events. it should be more than before!", len(chart_events))

    print(chart_events['readmit_label'].sum()) #confirm ratio positive to total
    print(chart_events.shape) # confirm ratio positive to total

    chart_events = chart_events.sort_values(['icustay_id', 'hours_from_beginning', 'category_num'])

    with open('chart_events_df.pickle', 'wb') as f:
        pickle.dump(chart_events, f)

    ############### VI. Collect ICD-9 data and one-hot encode it on a per-icustay basis

    #create dict of unique icd9 codes with arbitrary labels
    unique_icd9s = dict((code, num) for num, code in enumerate(set(icd_9['icd9_code'].values)))

    #add col of icustay_id to icd_9
    icd_9 = icd_9.merge(icu_stays[['hadm_id', 'icustay_id']], left_on='hadm_id', right_on='hadm_id')
    icd_9['icd_label'] = icd_9['icd9_code'].map(unique_icd9s)

    #check if icd_9's icustay_ids are in chart_events and vice versa
    #if icd9's ids not in chart_events, drop
    #if chart_events's ids not in icd9...drop?
    # icd9_not_ce = icd_9.loc[~icd_9['icustay_id'].isin(chart_events['icustay_id'])]['icustay_id'].unique()
    # ce_not_icd9 = chart_events.loc[~chart_events['icustay_id'].isin(icd_9['icustay_id'])]['icustay_id'].unique()
    # in_both = icd_9.loc[icd_9['icustay_id'].isin(chart_events['icustay_id'])]['icustay_id'].unique()

    # print('icd9 not ce:\n', icd9_not_ce)
    # print('ce not idc9:\n', ce_not_icd9)
    # print('in both: ', in_both)

    icd9_in_ce = icd_9.loc[icd_9['icustay_id'].isin(chart_events['icustay_id'])]

    one_hot = icd9_in_ce.sort_values('icustay_id').groupby('icustay_id')['icd9_code'].apply(list).reset_index().set_index('icustay_id')

    #print(one_hot.head(10))

    mlb = MultiLabelBinarizer()

    res = pd.DataFrame(mlb.fit_transform(one_hot['icd9_code']),
                    columns=mlb.classes_,
                    index=one_hot.index)

    #print(res.transpose().head(10))

    dict_test = res.transpose().to_dict(orient='list')
    #print(dict_test)

    #assert that the sorted icustay values in the dictionary are exactly the same as the sorted list from chart_events
    assert len(dict_test) == len(chart_events['icustay_id'].unique())
    assert list(dict_test.keys()) == chart_events['icustay_id'].unique().tolist()

    with open('icd9_per_icustay.pickle', 'wb') as f:
        pickle.dump(list(dict_test.values()), f)

    del icd_9
    del icu_stays
    del patients

    ############### VII. Create lists of pids, evids and events

    # 1. Create pids, evids, and readmit_label pickles

    with open('icu_stays.pickle', 'wb') as f:
        pickle.dump(chart_events['icustay_id'].to_list(), f)

    with open('evids.pickle', 'wb') as f:
        pickle.dump(chart_events['row_id'].to_list(), f)

    with open('readmit_labels.pickle', 'wb') as f:
        pickle.dump(chart_events['readmit_label'].to_list(), f)

    # 2. Create categories pickle
    with open('categories.pickle', 'wb') as f:
        pickle.dump(category_dict, f)

    # 3. Create events pickle using events_to_list
    # conduct this using events_to_list.py to avoid memory issues

    end_time = time.monotonic()
    print(datetime.timedelta(seconds=end_time - start_time))


