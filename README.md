# Chemotherapy + Actigraphy

This is code for the analysis of sleep and circadian rhythms in a dataset of chemotherapy timings and co-recorded actigraphy. 

These data files are VERY BIG, so processing works best on a machine with at least 400 GB of memory. If you want to carry out the processing on a Mac, use `raw_data_processing.py`; adjust that file as needed for Windows.

Files are reduced and stored as snapshots, which can be used to generate actograms.


The relevant files are: 
- `generate_model_snapshots.py`: Run the actigraphy data in chunks; save model output to `model_snapshots`
- `build_people_dictionary.py`: Create an instance of the Participant class for all people and saves it to a pickle file
- `validate_data.py`: Simple data validation to report subject counts and other qualities we wish to assert about the data
- `make_study_explainer.py`: Makes a simple illustration of the way the study was carried out
- `make_histogram_of_times.py`: Generates a histogram of DLMO, dosing time, and amplitude

Various utilities and helper functions are in `utils.py`
