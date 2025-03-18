import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import times_as_hours, get_valid
from scipy.signal import find_peaks
plt.rcParams["font.family"] = "Arial"


def make_histogram():
    with open('people_dict.pkl', 'rb') as file:
        people_dict = pickle.load(file)

    label_font_size = 14
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    all_people_times = []
    for person_id, participant in people_dict.items():
        treatment_times = participant.treatment_times
        all_people_times.extend(treatment_times)

    all_people_times = times_as_hours(all_people_times)

    # All treatment times
    plt.hist(all_people_times, bins=30, edgecolor='black',
             color=colors[0], alpha=0.7)
    plt.xlabel('Treatment Time', fontsize=label_font_size)
    plt.ylabel('Count', fontsize=label_font_size)
    ax = plt.gca()
    xticks = np.arange(6, 23, 2)
    xtick_labels = ['{:02d}:00'.format(int(x)) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("figures/histogram_of_dose_times.png", dpi=300)

    all_dlmos = []
    all_amplitudes = []
    all_average_dlmos = []
    all_alternative_dlmos = []

    for person_id, participant in people_dict.items():

        simulation_state = participant.simulation_state

        dlmos_for_person = []
        alternative_dlmos_for_person = []
        amplitudes_for_person = []

        for t, simulation_result in zip(participant.simulation_time, simulation_state):

            t, amplitude, phase = get_valid(t, simulation_result)

            if amplitude is None:
                continue

            amplitudes_for_person.extend(amplitude)
            distance_from_pi = -np.abs(phase - np.pi) + 1
            cbt_min_time_indices, _ = find_peaks(distance_from_pi, 0.9)
            cbt_min_times_in_seconds = t[cbt_min_time_indices]

            # Another way of calculating, for debugging.
            alternative_dlmo = np.mod(
                cbt_min_times_in_seconds, 3600 * 24) / (3600 * 24) * 24 - 7

            oriented_dlmos = np.mod(np.array(participant.dlmos) + 12, 24) - 12

            dlmos_for_person.extend(oriented_dlmos)
            alternative_dlmos_for_person.extend(alternative_dlmo)

        if len(dlmos_for_person) == 0:
            continue

        all_average_dlmos.append(np.mean(dlmos_for_person))
        all_dlmos.extend(dlmos_for_person)
        all_amplitudes.extend(amplitudes_for_person)
        all_alternative_dlmos.extend(alternative_dlmos_for_person)

    print("Average DLMO, weighting everyone equally:", np.mean(all_dlmos))
    print("Average Alternative Calculation DLMO:",
          np.mean(all_alternative_dlmos))

    plt.figure()
    plt.hist(all_dlmos, bins=30, edgecolor='black', color=colors[1], alpha=0.7)
    plt.xlabel('pDLMO Time', fontsize=label_font_size)
    plt.ylabel('Count', fontsize=label_font_size)
    ax = plt.gca()
    xticks = np.arange(-6, 3, 2)
    xtick_labels = ['18:00', '20:00', '22:00',
                    '00:00', '02:00']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("figures/histogram_of_dlmos.png", dpi=300)

    # Average DLMO by person
    plt.figure()
    plt.hist(all_average_dlmos, bins=30,
             edgecolor='black', color=colors[1], alpha=0.7)
    plt.xlabel('pDLMO Time', fontsize=label_font_size)
    plt.ylabel('Count', fontsize=label_font_size)
    ax = plt.gca()
    xticks = np.arange(-6, 3, 2)
    xtick_labels = ['18:00', '20:00', '22:00',
                    '00:00', '02:00']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("figures/histogram_of_normalized_dlmos.png", dpi=300)

    # Amplitude
    plt.figure()
    plt.hist(all_amplitudes, bins=30, edgecolor='black',
             color=colors[2], alpha=0.7)
    plt.xlabel('Model Amplitude', fontsize=label_font_size)
    plt.ylabel('Count', fontsize=label_font_size)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("figures/histogram_of_amplitudes.png", dpi=300)


if __name__ == '__main__':

    make_histogram()
