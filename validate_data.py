import pickle
import numpy as np
from utils import *


def validate_data():
    with open('people_dict.pkl', 'rb') as file:
        loaded_people_dict = pickle.load(file)

    print("Total number of participants:", len(loaded_people_dict))

    had_treatment = 0
    alive_count = 0
    deceased_count = 0
    os_of_deceased = []

    for participant in loaded_people_dict.values():

        number_of_treatments = participant.treatment_times
        if len(number_of_treatments) > 0:
            had_treatment += 1

        if participant.status == 1:
            alive_count += 1
        elif participant.status == 2:
            deceased_count += 1
        else:
            print(participant)
        os_of_deceased.append(get_os_for_person(participant))

        for sim_time in participant.simulation_time:
            if not all(x < y for x, y in zip(sim_time[:-1], sim_time[1:])):
                print(
                    f"Simulation times are not monotonically increasing for participant {participant.id_number}")

        if int(participant.id_number) == 5094:
            assert (participant.survey_dict['Q1fsisev'] == 6.25)

        print(f"Number of simulation batches for person {participant.id_number}: ", len(
            participant.simulation_time))
    print("Number of participants who had at least one treatment:", had_treatment)
    print("Number of participants who are alive:", alive_count)
    print("Number of participants who are deceased:", deceased_count)
    print("Average OS of deceased:", np.nanmean(os_of_deceased))


if __name__ == '__main__':
    validate_data()
