import glob
import os
import re
import pickle
import numpy as np
from lco import SinglePopModel
from utils import *
import time
import matplotlib.pyplot as plt


def zeitgeber_transform(counts):
    return 10 * counts


def make_model_snapshots():
    directory = "snapshots"
    pickle_files = glob.glob(os.path.join(directory, '*.pkl'))
    pickle_files.sort(key=lambda x: int(
        re.findall(r'\d+', os.path.basename(x))[0]))

    # For debugging
    # pickle_files = ["snapshots/0029_binned_data.pkl"]

    gap_threshold = 31  # Seconds
    for filename in pickle_files:
        if filename[10] != "5":  # Skip the 9000s—we're focused on the 5000s right now
            continue
        data_read = read_pickle_file(filename)
        time = data_read["binned_time"]
        counts = data_read["binned_sum"]
        if len(counts) > 0:
            counts = zeitgeber_transform(counts)
            process_and_save(time, counts, gap_threshold, filename)


def find_segments(time_vector, gap_threshold):
    gaps = np.where(np.abs(np.diff(time_vector)) > gap_threshold)[0]
    segments = np.split(time_vector, gaps + 1)
    return segments


def process_and_save(time_vector, counts_vector, gap_threshold, filename):
    # Split the time and counts vectors based on the gap threshold
    time_segments = find_segments(time_vector, gap_threshold)
    counts_segments = np.split(counts_vector, np.where(
        np.abs(np.diff(time_vector)) > gap_threshold)[0] + 1)

    results = []
    all_dlmos = []
    chunk_count = 0
    zeros_prepended_time_segments = []

    # Process each segment and store results
    for time_chunk, counts_chunk in zip(time_segments, counts_segments):
        if len(time_chunk) > 0:

            time_segment_with_zeros, result, dlmos = run_model(
                time_chunk, counts_chunk)
            zeros_prepended_time_segments.append(time_segment_with_zeros)
            first_nan_index = np.where(np.isnan(result[0, :]))[0][0] if np.isnan(
                result[0, :]).any() else np.nan

            if ~np.isnan(first_nan_index):
                print("NAN ENCOUNTERED WEE-OO WEE-OO NAN ENCOUNTERED")
                print(
                    "Don't just me for this print statement; you do what you have to do when you're debugging.")

            plt.close()
            plt.plot(result[0, :] * np.cos(result[1, :]))
            plt.savefig(
                f"output/{filename.split('/')[1]}_{chunk_count}.png", dpi=300)
            results.append(result)
            all_dlmos.extend(dlmos)
            chunk_count += 1

    save_name = "model_snapshots/" + filename.split('/')[1]

    with open(save_name, 'wb') as f:
        pickle.dump((time_segments, counts_segments,
                    zeros_prepended_time_segments, results, all_dlmos), f)

    print("Results saved to", save_name)


def run_model(timestamps, zeitgeber):
    initial_condition = np.array([0.6, phase_ic_guess(0), 0.0])

    model = SinglePopModel()

    min_time = np.min(timestamps) - np.mod(np.min(timestamps),
                                           SECONDS_PER_HOUR * HOURS_PER_DAY)

    # Generate timestamps from midnight to the first timestamp with 30-second intervals
    additional_timestamps = np.arange(min_time, np.min(timestamps), 30)
    additional_zeitgeber = np.zeros_like(additional_timestamps)

    # Concatenate the additional timestamps and zeitgeber values with the original ones
    timestamps = np.concatenate((additional_timestamps, timestamps))
    zeitgeber = np.concatenate((additional_zeitgeber, zeitgeber))

    timestamps = (timestamps - min_time) / SECONDS_PER_HOUR

    zeitgeber[np.isnan(zeitgeber)] = 0

    sol = model.integrate_model(timestamps,
                                zeitgeber,
                                initial_condition)

    dlmos = model.integrate_observer(
        timestamps, zeitgeber, initial_condition, SinglePopModel.DLMOObs)

    return timestamps, sol, dlmos


if __name__ == "__main__":
    start_time = time.time()
    make_model_snapshots()
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")
