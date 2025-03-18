
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
from utils import *
import numpy as np
import pickle
from statsmodels.genmod.generalized_estimating_equations import GEE
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt


def prepare_data(people_dict, survey_type,  correction_type=CorrectionType.NONE):
    data = []

    # For each person...
    for person_id, person_data in people_dict.items():

        # Loop over all of their treatment rounds
        for i, round_num in enumerate(person_data.treatment_rounds):

            dosing_times = times_as_hours(person_data.treatment_times)

            if correction_type == CorrectionType.BY_AVERAGE:
                dosing_times = correct_dosings_by_dlmo(
                    person_data, correction_type)

            time_for_dose = dosing_times[i]

            # Get survey for after treatment
            # TODO: Also look at how they respond to treatment (e.g. want_pre=True - want_pre=False)
            survey_num = round_to_survey_number(round_num, want_pre=False)
            column_name = f"Q{survey_num}{survey_type}"
            baseline_column = f"Q1{survey_type}"

            # Why use average amplitude?
            # Some actigraphy files have lost the date information
            # As a result, we can't know for sure which day the actigraphy data is from
            # This affects 20% of the data or so, so it's a sizable problem
            # Instead we'll use their amplitude over all days

            average_amplitude = average_overall_amplitude_for_participant(
                person_data)
            if column_name in person_data.survey_dict:
                survey_response = person_data.survey_dict[column_name]

                data.append({
                    'id': person_id,
                    'month': round_num,
                    'response': survey_response,
                    'amplitude': average_amplitude,
                    'timing': time_for_dose,
                    'baseline': person_data.survey_dict[baseline_column]
                })

    df = pd.DataFrame(data)
    return df


def do_statistics(people_dict):

    for outcome in outcomes_of_interest:
        df = prepare_data(people_dict, outcome,
                          correction_type=CorrectionType.BY_AVERAGE)

        df['normalized_response'] = df['response'] - df['baseline']
        df = df.dropna(subset=['normalized_response', 'response',
                       'amplitude', 'timing', 'id', 'baseline'])

        df['timing_scaled'] = (
            df['timing'] - df['timing'].mean()) / df['timing'].std()
        df['amplitude_scaled'] = (
            df['amplitude'] - df['amplitude'].mean()) / df['amplitude'].std()

        df['timing_scaled_squared'] = df['timing_scaled'] ** 2

        # Mixed-effects model
        model = MixedLM.from_formula(
            'response ~ amplitude_scaled + timing_scaled + timing_scaled_squared + baseline',
            groups='id',
            data=df
        )
        result = model.fit()
        print(outcome)

        print(result.summary())
        print("\n\n")

        # # GEE for comparison
        # model = GEE.from_formula(
        #     'normalized_response ~ amplitude + timing', groups='id', data=df)
        # result = model.fit()
        # print(result.summary())


def do_survival_analysis(df):

    # Aggregate data for survival model
    survival_df = df.groupby('id').agg({
        'timing': 'mean',
        'amplitude': 'mean',
        'survival_time': 'first',
        'event': 'first'  # Binary: 1 = event occurred, 0 = censored
    })
    survival_df['timing_squared'] = survival_df['timing'] ** 2

    # Cox proportional hazards model
    cph = CoxPHFitter()
    cph.fit(survival_df, duration_col='survival_time', event_col='event')
    cph.print_summary()


if __name__ == "__main__":
    with open('people_dict.pkl', 'rb') as file:
        loaded_people_dict = pickle.load(file)

    do_statistics(loaded_people_dict)
