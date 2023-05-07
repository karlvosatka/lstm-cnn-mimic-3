import os
import pandas as pd
from sklearn.utils.random import sample_without_replacement

########################################################################################################################
#                                         MIMIC III DATASET SUB-SAMPLER
########################################################################################################################

RAND_SEED = 62368
SAMPLE_SIZE = 10_000
MIMIC_III_PATH = './MIMIC III/'
DATA_SUBSET_PATH = './M3_Subset/'

os.chdir('/') # NOTE Set to correct project directory

assert os.path.isdir(MIMIC_III_PATH), f"{MIMIC_III_PATH} directory is missing"
assert os.path.isfile(MIMIC_III_PATH + 'ICUSTAYS.csv'), "ICUSTAYS.csv file is missing"
assert os.path.isfile(MIMIC_III_PATH + 'CHARTEVENTS.csv'), "CHARTEVENTS.csv file is missing"

if not os.path.isdir(DATA_SUBSET_PATH):
    os.mkdir(DATA_SUBSET_PATH)

# STEP 1: RANDOMLY SAMPLE INDICES FROM ICUSTAYS


def lower_columns(df):
    df_cols = list(df)
    df_cols_lowercase = [col_name.lower() for col_name in df_cols]
    cols_dict = dict(zip(df_cols, df_cols_lowercase))

    df.rename(columns=cols_dict, inplace=True)


icu_stays = pd.read_csv(MIMIC_III_PATH + 'ICUSTAYS.csv', header=0, parse_dates=['INTIME', 'OUTTIME'])
lower_columns(icu_stays)

subset_idx = sample_without_replacement(icu_stays.shape[0], SAMPLE_SIZE, random_state=RAND_SEED)

# STEP 2: SUB-SAMPLE ICUSTAYS.CSV

icu_stays = icu_stays.loc[subset_idx]
icu_stays.to_csv(DATA_SUBSET_PATH + 'ICUSTAYS.csv', mode='w')

# STEP 3: SUB-SAMPLE CHARTEVENTS.CSV

CHUNK_SIZE = 10000000
CHART_EVENT_COLS = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']

if os.path.isfile(DATA_SUBSET_PATH + 'CHARTEVENTS.csv'):
    os.remove(DATA_SUBSET_PATH + 'CHARTEVENTS.csv')

with pd.read_csv(MIMIC_III_PATH + 'CHARTEVENTS.csv', usecols=CHART_EVENT_COLS, parse_dates=['CHARTTIME'], chunksize=CHUNK_SIZE) as reader:
    for i, chunk in enumerate(reader):
        lower_columns(chunk)
        chunk = chunk.loc[chunk['icustay_id'].isin(icu_stays['icustay_id'])]
        header = True if i == 0 else None

        chunk.to_csv(DATA_SUBSET_PATH + 'CHARTEVENTS.csv', header=header, mode='a')

# STEP 4: (OPTIONAL) LOWERCASE COLUMNS FOR REMAINING FILES

ALL_FILES_LOWER_COLUMNS = True
OTHER_FILES = ['PATIENTS.csv', 'DIAGNOSES_ICD.csv', 'D_ITEMS.csv']

if ALL_FILES_LOWER_COLUMNS:
    for filename in OTHER_FILES:
        df = None

        if filename == 'PATIENTS.csv':
            df = pd.read_csv(MIMIC_III_PATH + filename, header=0, parse_dates=['DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN'])
        else:
            df = pd.read_csv(MIMIC_III_PATH + filename, header=0)

        lower_columns(df)
        df.to_csv(DATA_SUBSET_PATH + filename, mode='w')
