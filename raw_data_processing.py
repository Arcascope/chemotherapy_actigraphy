import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib
import pickle


DELTA_T = 0.1
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
INVALID_DAY_THRESHOLD = 50


def read_all_for(id):
    # Specify the directory where the files are stored
    directory = 'data'

    # Initialize lists to hold combined data
    combined_time = []
    combined_amplitude = []
    print(os.path.join(directory, f'{id}*.csv'))

    # Loop through each file in the directory
    for filepath in glob.glob(os.path.join(directory, f'{id}*.csv')):
        # Read metadata
        metadata = pd.read_csv(filepath, nrows=8, header=None)

        days_str = metadata.iloc[2, 0]
        days = int(days_str.split()[1])

        hz_str = metadata.iloc[0, 0]
        hz_parts = hz_str.split()
        hz_index = hz_parts.index('Hz')
        hz = int(hz_parts[hz_index - 1])

        start_seconds = days * 24 * 3600

        # Read actual data, skipping metadata lines
        df = pd.read_csv(filepath, skiprows=9)
        x = df['Accelerometer X'].values
        y = df['Accelerometer Y'].values
        z = df['Accelerometer Z'].values

        total_seconds = len(x) * (1 / hz) + start_seconds
        time_vector = np.arange(start_seconds, total_seconds, 1 / hz)

        amplitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Append to combined arrays
        combined_time.extend(time_vector)
        combined_amplitude.extend(amplitude)

    binned_time, binned_sum = reduce_to_counts(
        combined_time, combined_amplitude)

    data_to_save = {
        'binned_time': binned_time,
        'binned_sum': binned_sum
    }
    filename = f'snapshots/{id}_binned_data.pkl'

    with open(filename, 'wb') as file:  # 'wb' is crucial for writing in binary mode
        pickle.dump(data_to_save, file)

    plot_actogram(id, binned_time, binned_sum)

    # 4. Plot the original data
    plt.figure(figsize=(10, 5))

    # Plot the binned cumulative sum data
    plt.step(binned_time, binned_sum, where='mid',
             label='Binned Sum', linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude and Cumulative Sum')
    plt.title('Amplitude Difference and Cumulative Sum')
    plt.legend()
    plt.savefig(f"output/{id}_time_series.png", dpi=300)
    plt.close()


def extended_nanmean(array_slice):
    if np.all(np.isnan(array_slice)):
        return np.nan
    else:
        return np.nanmean(array_slice)


def reduce_to_counts(combined_time, combined_amplitude):
    # 1. Calculate the numerical difference and take the absolute value
    amplitude_diff = np.abs(np.diff(combined_amplitude))

    # 2. Define the bin size in seconds
    bin_size_seconds = 30  # for example, change this to your desired bin size
    hz = 30  # This needs to be defined based on your data's sampling rate

    # Calculate the number of data points per bin
    points_per_bin = bin_size_seconds * hz

    # 3. Calculate the cumulative sum in bins
    binned_sum = np.add.reduceat(amplitude_diff, np.arange(
        0, len(amplitude_diff), points_per_bin))

    # Adjust the time vector for binned data
    # Start from second element due to diff
    binned_time = combined_time[1:][::points_per_bin]

    return binned_time, binned_sum


def plot_actogram(id, time, counts):

    num_days = int((np.max(time) - np.min(time)) //
                   (SECONDS_PER_HOUR * HOURS_PER_DAY))
    min_time = np.min(time)
    actogram = []

    # Create an actogram
    for i in range(num_days):
        activity_in_day = []
        start_of_day = min_time + HOURS_PER_DAY * SECONDS_PER_HOUR * i
        end_of_day = min_time + HOURS_PER_DAY * SECONDS_PER_HOUR * (i + 1)

        # Iterate over chunks of day
        for time_chunk in range(int(HOURS_PER_DAY / DELTA_T)):
            time_of_interest = start_of_day + time_chunk * DELTA_T * SECONDS_PER_HOUR
            steps_in_range = counts[
                (time >= time_of_interest) & (time < time_of_interest + DELTA_T * SECONDS_PER_HOUR)]
            if len(steps_in_range) > 0:
                ac_value = extended_nanmean(steps_in_range)
            else:
                ac_value = np.nan
            activity_in_day.append(ac_value)

        # If a day has no data (or is below a threshold), mark it as invalid
        if np.sum(activity_in_day) < INVALID_DAY_THRESHOLD or np.isnan(np.sum(activity_in_day)):
            actogram.append(np.ones_like(activity_in_day) * np.nan)
        else:
            actogram.append(activity_in_day)

    actogram_shifted = actogram[1:]
    actogram_shifted.append(actogram[0])
    actogram = np.hstack((np.array(actogram), np.array(actogram_shifted)))
    # Rescale and plot
    actogram = np.log(actogram + 1)
    cmap = matplotlib.colormaps["viridis"]
    cmap.set_bad(color='k')  # Show NaNs in different color

    plt.imshow(actogram, interpolation='none',
               aspect='auto', vmin=0, vmax=10, cmap=cmap)
    plt.savefig(f"output/{id}_actogram.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    directory = 'data'

    subject_ids = set()

    for filename in os.listdir(directory):
        if filename.endswith("RAW.csv"):  # Check if the filename ends with 'RAW.csv'
            # Split the filename by space and take the first part
            subject_id = filename.split()[0]
            subject_ids.add(subject_id)  # Add the subject ID to the set

    subject_ids = sorted(subject_ids)

    print(subject_ids)
    for subject_id in subject_ids:
        read_all_for(subject_id)
