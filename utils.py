import numpy as np
import pandas as pd
import pickle
import os
from scipy.signal import find_peaks
import enum

DELTA_T = 0.1
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
DAYS_TO_REMOVE_FOR_IC = 4
AVERAGE_DLMO_IN_POPULATION = 21.8


significant_outcomes = {'fsisevsig': "Fatigue Severity Index",
                        'hadsdepsig': "HADS Depression",
                        'hadsanxsig': "HADS Anxiety"}

outcomes_of_interest = {'fsisev': "Fatigue Severity Index",
                        # 'fsidis': "Fatigue disruptiveness", # Some of these have 2 in the name
                        'hadsdep': "HADS Depression",
                        'hadsanx': "HADS Anxiety",
                        'PSQTotal': "PSQI",
                        'PSSTotal': "Perceived Stress Scale",
                        # 'ipaqTotalMETS': "IPAQ Total METS (higher better)",
                        'IESITotal': "IES-R Intrusion",
                        'CIPNTotal': "CIPN Total",
                        'CIPNSens': "CIPN Sensory",
                        'CIPNMotor': "CIPN Motor",
                        'CIPNAuto': "CIPN Autonomic"}


# Survey to round mapping
# 0   Pre-surgery 0
# 1   Pre-chemo 1
# 2   Post-chemo 1
# 3   Pre-chemo 3
# 4   Post-chemo 3
# 5   Pre-chemo 6
# 6   Post-chemo 6
# 7   6-months after chemo
# 8   12 months after chemo

class CorrectionType(enum.Enum):
    NONE = 0
    BY_AVERAGE = 1
    INDIVIDUALLY = 2


def correct_dosings_by_dlmo(participant, correction_type):

    if correction_type == CorrectionType.NONE:
        return times_as_hours(participant.treatment_times)

    if correction_type == CorrectionType.BY_AVERAGE:
        dosing_times = times_as_hours(participant.treatment_times)

        adjustment_term = np.mean(participant.dlmos) - \
            AVERAGE_DLMO_IN_POPULATION
        dosing_times = [
            time - adjustment_term for time in dosing_times]
        return dosing_times

    if correction_type == CorrectionType.INDIVIDUALLY:
        raise NotImplementedError(
            "Don't use this function for individual corrections")


def average_dosing_time_for_participant(participant, correction_type):
    dosing_times = correct_dosings_by_dlmo(participant, correction_type)
    dosing_fractions_of_day = [time / 24 for time in dosing_times]
    return calculate_circular_mean_from_day_fraction(
        dosing_fractions_of_day)


def average_overall_amplitude_for_participant(participant):
    amplitudes = []

    simulation_times = participant.simulation_time
    # print(
    #     f"Running participant {participant.id_number} ")
    # Loop over all simulation blocks
    for k, simulation_time in enumerate(simulation_times):

        sim_time, amplitude, phase = get_valid(
            simulation_time, participant.simulation_state[k])

        if amplitude is not None:
            amplitudes.extend(amplitude)

    return np.mean(amplitudes)


def round_to_survey_number(round_num, want_pre):
    if round_num == 1 and want_pre:
        return 1
    if round_num == 1 and not want_pre:
        return 2
    if round_num == 3 and want_pre:
        return 3
    if round_num == 3 and not want_pre:
        return 4
    if round_num == 6 and want_pre:
        return 5
    if round_num == 6 and not want_pre:
        return 6
    return np.nan


def survey_to_round_number(survey_num, want_pre):
    if survey_num == 1 and want_pre:
        return 1
    if survey_num == 2 and not want_pre:
        return 1
    if survey_num == 3 and want_pre:
        return 3
    if survey_num == 4 and not want_pre:
        return 3
    if survey_num == 5 and want_pre:
        return 6
    if survey_num == 6 and not want_pre:
        return 6
    return np.nan


def read_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:  # 'rb' is crucial for reading in binary mode
            data = pickle.load(file)
    else:
        return None
    return data


def phase_ic_guess(time_of_day: float):
    time_of_day = np.fmod(time_of_day, 24.0)

    # Wake at 7 am after 8 hours of sleep, state at 00:00
    psi = 1.65238233

    # Convert to radians, add to phase
    psi += time_of_day * np.pi / 12
    return psi


def get_os_for_person(participant):
    if participant.status == 2:
        return participant.last_contact_day
    return np.nan


def get_valid(t, simulation_result):
    start_index = int(DAYS_TO_REMOVE_FOR_IC * 24 * (3600 / 30))

    amplitude = simulation_result[0, start_index:]
    phase = np.mod(simulation_result[1, start_index:], 2*np.pi)
    t = t[start_index:]

    # Remove invalid indices
    invalid_indices = (amplitude > 1.1) | (amplitude < 0)
    t = t[~invalid_indices]
    amplitude = amplitude[~invalid_indices]
    phase = phase[~invalid_indices]

    # Ensure the length of t, amplitude, and phase is a multiple of 24 hours
    length = len(t)
    remainder = length % (3600 * 24 / 30)
    if remainder != 0:
        t = t[:-int(remainder)]
        amplitude = amplitude[:-int(remainder)]
        phase = phase[:-int(remainder)]

    if len(t) == 0:
        # print("No valid data!")
        return None, None, None
    if (max(t) - min(t)) / (3600 * 24) < 0.9:
        # print("Significantly less than a day of data!")
        # print((max(t) - min(t)) / (3600 * 24))
        return t, amplitude, phase

    return t, amplitude, phase


def calculate_circular_mean_from_day_fraction(day_fraction):
    # Ignore zeros
    times = np.array(day_fraction)
    times = times[times > 0]

    radians = [(time) * 2 * np.pi for time in times]
    return calculate_circular_mean_radians_to_hours(radians)


def calculate_circular_mean_from_raw_times(times):
    # Convert times to radians
    times = time_to_days(times)
    # Ignore zeros
    times = np.array(times)
    times = times[times > 0]

    radians = [(time) * 2 * np.pi for time in times]
    return calculate_circular_mean_radians_to_hours(radians)


def time_to_days(times):
    return [((time.hour * 3600 + time.minute * 60 +
              time.second) / 86400) for time in times]


def time_to_hours(times):
    times = time_to_days(times)
    return [time * 24 for time in times]


def times_as_hours(treatment_times):
    new_treatment_times = []
    for time in treatment_times:
        if time != '00:00:00':
            new_treatment_times.append(time)
        else:
            new_treatment_times.append(np.nan)
    return [pd.to_timedelta(time).total_seconds() / 3600 for time in new_treatment_times]


def calculate_circular_mean_radians_to_hours(radians):
    mean_angle = np.arctan2(np.mean(np.sin(radians)),
                            np.mean(np.cos(radians)))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    # Convert circular mean to hours after midnight
    mean_time = (mean_angle / (2 * np.pi)) * 24
    return mean_time


def get_dlmo_from_phase(t, phase):
    distance_from_pi = -np.abs(phase - np.pi) + 1
    cbt_min_time_indices, _ = find_peaks(distance_from_pi, 0.9)
    cbt_min_times_in_seconds = t[cbt_min_time_indices]

    # Another way of calculating, for debugging.
    alternative_dlmo = 1 + (np.mod(
        cbt_min_times_in_seconds, 3600 * 24) / (3600 * 24) * 24 - 7) / 24

    if len(alternative_dlmo) > 7 - DAYS_TO_REMOVE_FOR_IC:
        # Try to do DLMO on the exact day of infusion. This assumes you've trimmed off DAYS_TO_REMOVE_FOR_IC
        return alternative_dlmo[7 - DAYS_TO_REMOVE_FOR_IC - 1]
    else:
        return alternative_dlmo[0]
    # return calculate_circular_mean_from_day_fraction(alternative_dlmo)


class Participant:
    def __init__(self,
                 id_number,
                 status,
                 last_contact_day,
                 survey_dict,
                 treatment_rounds,
                 treatment_days,
                 treatment_times,
                 counts_time,
                 actigraphy_counts,
                 simulation_time,
                 simulation_state,
                 dlmos):
        self.id_number = id_number
        self.status = status
        self.last_contact_day = last_contact_day
        self.survey_dict = survey_dict
        self.treatment_rounds = treatment_rounds
        self.treatment_days = treatment_days
        self.treatment_times = treatment_times
        self.counts_time = counts_time
        self.actigraphy_counts = actigraphy_counts
        self.simulation_time = simulation_time
        self.simulation_state = simulation_state
        self.dlmos = dlmos

    def __repr__(self):
        return (f"Participant(survey_dict={self.survey_dict}, "
                f"treatment_days={self.treatment_days}, "
                f"treatment_times={self.treatment_times}, "
                f"simulation_time={self.simulation_time}, "
                f"simulation_state={self.simulation_state})")
