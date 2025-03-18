
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from utils import *
import numpy as np
plt.rcParams['font.family'] = 'Arial'


def make_outcome_plots(people_dict):

    correction_type = CorrectionType.NONE
    drug = ""  # Use this to filter for only short half-life drugs; e.g. with carbo
    outcomes = outcomes_of_interest.keys()
    want_normalize = False

    for outcome in outcomes:
        want_pre = False
        overall_survival = []
        average_time_of_day = []

        outcomes_dosing_time = []
        times_of_dosing_times = []

        amplitude_values = []
        outcomes_amplitude = []

        counts_values = []
        outcomes_counts = []

        for person_id, participant in people_dict.items():
            if drug in str(participant.survey_dict['FU12CHEMDRUGS']).lower():

                overall_survival.append(get_os_for_person(participant))

                dosing_times = correct_dosings_by_dlmo(
                    participant, correction_type)

                average_dosing_time = average_dosing_time_for_participant(
                    participant, correction_type)
                average_time_of_day.append(average_dosing_time)

                rounds = participant.treatment_rounds

                if want_normalize:
                    # If we use Q0 here instead of Q1, we lose all the 5000s
                    base_column = f"Q1{outcome}"
                    base_column_value = participant.survey_dict[
                        base_column] if base_column in participant.survey_dict else np.nan
                else:
                    base_column_value = 0

                # Loop over all rounds
                average_survey_response = 0
                counts_for_survey = 0
                for i, round_num in enumerate(rounds):

                    time = dosing_times[i]
                    survey_num = round_to_survey_number(round_num, want_pre)
                    column_name = f"Q{survey_num}{outcome}"

                    if column_name in participant.survey_dict:
                        survey_response = participant.survey_dict[column_name] - \
                            base_column_value

                        times_of_dosing_times.append(time)
                        outcomes_dosing_time.append(survey_response)
                        average_survey_response += survey_response
                        counts_for_survey += 1

        plot_outcome(average_time_of_day, overall_survival,
                     'Average Dosing Time',
                     'Overall Survival (days)',
                     'Overall Survival vs. Average Dosing Time',
                     'drug_plot_average_time.png',
                     'Average Time')

        plot_outcome(amplitude_values, outcomes_amplitude,
                     'Amplitude',
                     outcomes_of_interest[outcome],
                     "",
                     f'{outcome}_vs_amplitude.png',
                     'Amplitude', want_linear=True)

        plot_outcome(counts_values, outcomes_counts,
                     'Counts',
                     outcome,
                     f'{outcome} vs. Counts',
                     f'{outcome}_vs_counts.png',
                     'Counts', want_linear=True)

        plot_outcome(times_of_dosing_times, outcomes_dosing_time,
                     'Dosing time',
                     outcomes_of_interest[outcome],
                     "",
                     f'{outcome}_vs_time.png',
                     'Time')


def plot_outcome(x_data, y_data, x_label, y_label, title, filename, label, want_linear=False):
    # Remove NaN values from x_data and y_data
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]

    mako_colors = sns.color_palette("mako", 5)

    if len(x_data) > 0:
        if want_linear:
            coefficients = np.polyfit(x_data, y_data, 1)
            polynomial = np.poly1d(coefficients)
        else:
            coefficients = np.polyfit(x_data, y_data, 2)
            polynomial = np.poly1d(coefficients)

        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = polynomial(x_fit)

        plt.plot(x_fit, y_fit, color=mako_colors[3],
                 label=f'{"Quadratic" if not want_linear else "Linear"} Fit ({label})')

        plt.scatter(x_data, y_data,
                    label=f'Data Points ({label})', color=mako_colors[0])
        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.gca().text(1, 0.95, f'$N=${len(x_data)}', transform=plt.gca().transAxes,
                       fontsize=33, verticalalignment='top', horizontalalignment='right')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig("output/" + filename, dpi=300)

        plt.close()


if __name__ == "__main__":
    with open('people_dict.pkl', 'rb') as file:
        loaded_people_dict = pickle.load(file)

    make_outcome_plots(loaded_people_dict)
