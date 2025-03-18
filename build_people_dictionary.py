import pandas as pd
import pickle
import os
import numpy as np
from utils import *
from scipy.stats import f
import matplotlib.pyplot as plt


def build_people():
    people_dict = {}

    # Read PRO data
    pro_path = 'PROData.xlsx'
    df = pd.read_excel(pro_path)

    pro_dict = {}
    for _, row in df.iterrows():
        person_id = row['ID']
        person_data = row.drop(labels=['ID']).to_dict()
        pro_dict[person_id] = person_data

    # Read Infusion Timing data
    infusion_path = 'ChemoInfusionDaysTimes.xlsx'
    df_infusion = pd.read_excel(infusion_path)

    infusion_dict = {}
    for _, row in df_infusion.iterrows():
        person_id = row['ID']
        if person_id not in infusion_dict:
            infusion_dict[person_id] = [[], [], []]
        infusion_dict[person_id][0].append(row['ROUND'])
        infusion_dict[person_id][1].append(row['FUSDAY'])
        infusion_dict[person_id][2].append(row['FUSTIME'])

    # Read Response data
    response_path = 'ResponseData.xlsx'
    df = pd.read_excel(response_path)

    key_count = 0
    for row_index, row in df.iterrows():
        person_id = row['ID']
        # Only use 5000s
        if person_id < 5000:
            continue

        print(person_id)
        if np.isnan(person_id):
            print(row_index)
            continue
        status = row['STATUS']
        last_contact_day = row['LAST_CONTACT_DAY']
        person_id = int(person_id)

        person_id_padded = str(person_id).zfill(4)

        model_file = f"model_snapshots/{person_id_padded}_binned_data.pkl"

        response = read_pickle_file(model_file)
        if response is not None:
            time_segments, counts_segments, simulation_time, result, dlmos = response
        else:
            time_segments, counts_segments, simulation_time, result, dlmos = [], [], [], [], []

        # Remove IC dlmos
        if len(dlmos) > DAYS_TO_REMOVE_FOR_IC:
            dlmos = dlmos[DAYS_TO_REMOVE_FOR_IC:]
        else:
            dlmos = []

        infusion_values = infusion_dict[person_id] if person_id in infusion_dict else [
            [], [], []]

        participant = Participant(id_number=row['ID'],
                                  status=status,
                                  last_contact_day=last_contact_day,
                                  survey_dict=pro_dict[person_id],
                                  treatment_rounds=infusion_values[0],
                                  treatment_days=infusion_values[1],
                                  treatment_times=infusion_values[2],
                                  simulation_time=simulation_time,
                                  counts_time=time_segments,
                                  actigraphy_counts=counts_segments,
                                  simulation_state=result,
                                  dlmos=np.mod(dlmos, 24))

        if any(10 < dlmo < 12 for dlmo in participant.dlmos):
            print(
                f"Warning: Participant {person_id} has a DLMO between 10 and 12.")

        key_count += 1
        if person_id in people_dict.keys():
            print(f"Duplicate key detected: {person_id}")
        people_dict[person_id] = participant

    print(f"Number of participants: {key_count}")
    print(f"Number of participants (unique): {len(people_dict)}")

    with open('people_dict.pkl', 'wb') as file:
        pickle.dump(people_dict, file)


if __name__ == '__main__':
    build_people()
