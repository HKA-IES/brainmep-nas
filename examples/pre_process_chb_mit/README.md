# Example - Pre-process CHB-MIT Dataset
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

This example demonstrates the pre-processing of the CHB-MIT Scalp EEG Dataset
and the creation of a Dataset object to manipulate the processed data. Processed data
is available both as time-series data (appropriate for CNN or RNN inference) and as
features (appropriate for decision trees ensemble).

Note: the full CHB-MIT Scall EEG Data requires 42.6 GB of memory space. The processed 
time series data requires 3.82 GB of memory space.

## Requirements
- CHB-MIT Scalp EEG Database, [link](https://physionet.org/content/chbmit/1.0.0/).
- Package [MNE-Python](https://mne.tools/stable/index.html). 
- Python 3.12 (example not tested with older versions)

## Usage
To process time series data, run
```
python process_time_series.py -i {chb_mit_directory} -o {desired_output_directory}
```
To process features, run
```
python process_features.py -i {processed_time_series} -o {desired_output_directory}
```

## Time-series processing
Pre-processing is done as described in [1], with missing information acquired
through personal communication with Thorir Ingolfsson.

The dataset is the CHB-MIT Scalp EEG Database and all patients are used. 
Only 4 channels situated in the temporal region are used: F7-T7, T7-P7, F8-T8, 
and T8-P8. A 4 second window is used, which corresponds to 1024 time samples. 
The DC offset and high-frequency noise are removed using a bandpass Butterworth
filter of order 5 with a cut-off frequency of 0.5-50 Hz. The training data is 
composed of non-overlaping windows, whereas test data is split using a sliding 
window of 2 seconds[2].

References: 
[1] T. M. Ingolfsson et al., “EpiDeNet: An Energy-Efficient 
Approach to Seizure Detection for Embedded Systems.” arXiv, Aug. 28, 2023. 
Accessed: Sep. 20, 2023. [Online]. Available: http://arxiv.org/abs/2309.07135
[2] P. Busia et al., “EEGformer: Transformer-Based Epilepsy Detection on Raw 
EEG Traces for Low-Channel-Count Wearable Continuous Monitoring Devices,” 
in 2022 IEEE Biomedical Circuits and Systems Conference (BioCAS), Oct. 2022, 
pp. 640–644. doi: 10.1109/BioCAS54905.2022.9948637.

## Features processing
Features processing is done based on processed time-series. The following 
features are calculated for the four channels:
- variance
- skewness
- kurtosis
- median absolute deviation
- line length
- maximum
- max power
- mean power
- power variance
- theta band power
- beta band power
- gamma band power
- epi index.
