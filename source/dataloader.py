# Adapted with reference to HW3_RNN from CS598: Deep Learning for Healthcare, University of Illinois Urbana Champaign

import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# pids_f = open('../data/pids.pickle', 'rb')
evids_f = open('../data/evids.pickle', 'rb')
readmit_labels_f = open('../data/readmit_labels.pickle', 'rb')
categories_f = open('../data/categories.pickle', 'rb')
events_f = open('../data/events.pickle', 'rb')
icd9_per_icustay_f = open('../data/icd9_per_icustay.pickle', 'rb')
icu_stays_f = open('../data/icu_stays.pickle', 'rb')

# pids = pickle.load(pids_f)
evids = pickle.load(evids_f)
readmit_labels = pickle.load(readmit_labels_f)
categories = pickle.load(categories_f)
events = pickle.load(events_f)
demo_features = pickle.load(icd9_per_icustay_f)
icu_stays = pickle.load(icu_stays_f)

# pids_f.close()
evids_f.close()
readmit_labels_f.close()
categories_f.close()
events_f.close()
icd9_per_icustay_f.close()
icu_stays_f.close()

class CustomDataset(Dataset):
    
    def __init__(self, events, demo_features, labels):
        self.x = events
        self.feats = demo_features
        self.y = labels
        self.shape = self.dims()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.feats[index], self.y[index]
        
    def dims(self):
        events = self.x
        num_patients = len(events)
        num_hours = [len(patient) for patient in events]
        num_categories = [len(hours) for patient in events for hours in patient]
        num_events = [len(category) for patient in events for hours in patient for category in hours]

        max_num_hours = max(num_hours)
        max_num_categories = max(num_categories)
        max_num_events = max(num_events)

        return num_patients, max_num_hours, max_num_categories, max_num_events

dataset = CustomDataset(events, demo_features, readmit_labels)

def collate_fn(data):
    events, features, labels = zip(*data)
    batch_size = len(events)
    _ , max_num_hours, max_num_categories, max_num_events = dataset.shape

    y = torch.tensor(labels, dtype=torch.long)
    # y = torch.zeros((len(labels), 2), dtype=torch.long)
    # for i_label, label in enumerate(labels):
    #    y[i_label, label] = 1

    x = torch.full((batch_size, max_num_hours, max_num_categories, max_num_events), 0.0, dtype=torch.float)
    masks = torch.zeros((batch_size, max_num_hours, max_num_categories, max_num_events), dtype=torch.bool)
    for i_patient, patient in enumerate(events):
        for j_hour, hour in enumerate(patient):
            for k_category, category in enumerate(hour):
                for l_event in range(len(category)):
                    x[i_patient][j_hour][k_category][l_event] = category[l_event]
                    masks[i_patient][j_hour][k_category][l_event] = True

    feats = torch.tensor(features, dtype=torch.long)

    return x, feats, masks, y


def load_data(train_sampler, val_sampler, collate_fn=collate_fn, batch_size=10):
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, val_loader
