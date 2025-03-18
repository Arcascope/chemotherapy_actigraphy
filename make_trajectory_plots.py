
import pingouin as pg
import pickle
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mako_colors = sns.color_palette("mako", 5)


def make_trajectory_plots(people_dict):

    for outcome in outcomes_of_interest.keys():
        all_people_trajectories = []
        all_time_trajectories = []
        all_round_trajectories = []
        want_normalize = True
        want_pre = False
        correction_type = CorrectionType.NONE
        drug = ""

        for person_id, participant in people_dict.items():
            if drug in str(participant.survey_dict['FU12CHEMDRUGS']).lower():
                # all_participant_ids.append(person_id)
                rounds = participant.treatment_rounds
                dosing_times = correct_dosings_by_dlmo(
                    participant, correction_type)

                person_trajectory = []
                time_trajectory = []
                round_trajectory = []

                # Loop over all rounds
                for i, round_num in enumerate(np.arange(1, 7, 1)):

                    if round_num not in rounds:
                        person_trajectory.append(np.nan)
                        time_trajectory.append(np.nan)
                        round_trajectory.append(round_num)
                        continue

                    time = dosing_times[i]
                    survey_num = round_to_survey_number(round_num, want_pre)
                    column_name = f"Q{survey_num}{outcome}"

                    if want_normalize:
                        # If we use Q0 here, we lose all the 5000s, as they do not have a Q0
                        # e.g. base_column = f"Q0{outcome}"
                        # Instead, normalize by Q1.

                        base_column = f"Q1{outcome}"
                        base_column_value = participant.survey_dict[
                            base_column] if base_column in participant.survey_dict else np.nan
                    else:
                        base_column_value = 1

                    # If the column exists in the survey...
                    if column_name in participant.survey_dict:
                        person_trajectory.append(
                            participant.survey_dict[column_name] - base_column_value)
                        time_trajectory.append(
                            time)
                        round_trajectory.append(round_num)
                    else:
                        person_trajectory.append(np.nan)
                        time_trajectory.append(time)
                        round_trajectory.append(round_num)

                if want_normalize:
                    person_trajectory.insert(0, 0)
                    time_trajectory.insert(0, np.nan)
                    round_trajectory.insert(0, 0)

                all_people_trajectories.append(person_trajectory)
                all_time_trajectories.append(time_trajectory)
                all_round_trajectories.append(round_trajectory)

        group_1 = []
        group_2 = []

        for i, person_trajectory in enumerate(all_people_trajectories):

            time_trajectory = all_time_trajectories[i]
            round_trajectory = all_round_trajectories[i]

            enough_threshold = 0.5
            morning_count = sum(6 <= time <= 11 for time in time_trajectory)

            evening_count = sum(14 < time <= 24 or 0 <= time <
                                6 for time in time_trajectory)

            midday_count = 0  # Currently ignoring midday dosers

            total_count = sum(~np.isnan(time_trajectory))

            if total_count == 0:
                continue
            if morning_count / total_count > enough_threshold:
                time_period = 'morning'
            elif midday_count / total_count > enough_threshold:
                time_period = 'midday'
            elif evening_count / total_count > enough_threshold:
                time_period = 'evening'
            else:
                continue

            time_period_colors = {
                'morning': 'orange',
                'midday': 'green',
                'evening': 'purple'
            }

            if time_period == 'morning':
                group_1.append(person_trajectory)
            elif time_period == 'evening':
                group_2.append(person_trajectory)

            plt.plot(round_trajectory,
                     person_trajectory,
                     'o--',
                     color=time_period_colors[time_period],
                     alpha=0.4)

        plt.title(f"{outcome} vs. Time of Day")
        plt.savefig(f"output/{outcome}_vs_time_of_day.png")
        plt.close()

        print("Shape of group 1:", np.array(group_1).shape)
        print("Shape of group 2:", np.array(group_2).shape)

        group_1_average = np.nanmean(np.array(group_1), axis=0)
        group_2_average = np.nanmean(np.array(group_2), axis=0)
        group_1_sem = np.nanstd(np.array(group_1), axis=0) / \
            np.sqrt(np.sum(~np.isnan(np.array(group_1)), axis=0))
        group_2_sem = np.nanstd(np.array(group_2), axis=0) / \
            np.sqrt(np.sum(~np.isnan(np.array(group_2)), axis=0))

        valid_indices = ~np.isnan(group_1_average)
        group_1_average = group_1_average[valid_indices]
        group_1_sem = group_1_sem[valid_indices]

        valid_indices = ~np.isnan(group_2_average)
        group_2_average = group_2_average[valid_indices]
        group_2_sem = group_2_sem[valid_indices]
        plt.rcParams['font.family'] = 'Arial'
        plt.errorbar(range(len(group_1_average)), group_1_average,
                     yerr=group_1_sem, fmt='o--', color=mako_colors[4], label='>50% Early Infusion')
        plt.errorbar(range(len(group_2_average)), group_2_average,
                     yerr=group_2_sem, fmt='o--', color=mako_colors[1], label='>50% Late Infusion')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.xlabel("Survey", fontsize=20)
        plt.ylabel(outcomes_of_interest[outcome], fontsize=20)

        plt.xticks(ticks=[0, 1, 2, 3], labels=[
                   '1', '2', '4', '6'], fontsize=16)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"output/{outcome}_vs_time_of_day_averaged.png", dpi=300)
        plt.close()
        # plt.show()

        data = {
            'subject': [],
            'group': [],
            'time': [],
            'value': []
        }

        print(outcome)
        group_1_count = 0
        for idx, trajectory in enumerate(group_1):
            for time, value in enumerate(trajectory):
                if np.isnan(value):
                    continue
                group_1_count += 1
                data['subject'].append(idx)
                data['group'].append('A')
                data['time'].append(time)
                data['value'].append(value)

        for idx, trajectory in enumerate(group_2):
            for time, value in enumerate(trajectory):
                if np.isnan(value):
                    continue

                data['subject'].append(idx + group_1_count)
                data['group'].append('B')
                data['time'].append(time)
                data['value'].append(value)

        df = pd.DataFrame(data)

        from statsmodels.regression.mixed_linear_model import MixedLM

        print(df.isnull().sum())  # Shows the count of NaNs per column

        model = MixedLM.from_formula(
            'value ~ group * time', groups='subject', data=df)
        result = model.fit()
        print(result.summary())


if __name__ == "__main__":
    with open('people_dict.pkl', 'rb') as file:
        loaded_people_dict = pickle.load(file)

    make_trajectory_plots(loaded_people_dict)
