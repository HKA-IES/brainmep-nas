# BrainMEP NAS - PUBLIC REPO

![Python](https://img.shields.io/badge/python-3.12-blue.svg)

This repository contains the public code used for the NAS process for the
Brain-MEP Project.

**Note: this package is a work in progress**

## Install it from the repository
Tested with Python 3.12.

```
pip install git+https://github.com/jonathanlarochelle/brainmep-nas
```

## Usage
The package *brainmepnas* consists in utility modules to faciliate the 
implementation of a NAS process for the Brain-MEP project. Example uses are
shown in examples/.

## To-do
- [ ] Metrics calculations
  - [x] AccuracyMetrics - calculates inference accuracy metrics like ROC-AUC, 
sensitivity, and false detections per hour.
  - [x] HardwareMetrics - calculates hardware metrics like energy and latency 
per inference.
    - [x] MltkHardwareMetrics - HardwareMetrics implementation using the 
Silicon Labs MLTK Model Profiler.
    - [x] TestbenchHardwareMetrics - HardwareMetrics implementation using the 
BrainMEP Seizure Detection Testbench.
    - [ ] RandomForestHardwareMetrics - HardwareMetrics implementation to 
estimate or measure the performance of random forest classifiers.
  - [x] CombinedMetrics - calculates metrics which combine accuracy and hardware
values like the combined energy for inferences and false detections per hour.
- [ ] Dataset utilities
  - [ ] Dataset class
- [ ] Study utilities
  - [ ] Script for study creation
  - [ ] Script for running a study
  - [ ] Script for running a single trial
  - [ ] Script for fully training best trials from study
- [ ] Visualization utilities
  - [ ] TBD

## Development
Please note that the goal of this package is not to offer a general framework 
for NAS. It was built around the needs of the Brain-MEP project and is designed
for this specific use-case.

Please do not hesitate to submit issues if you have found bugs or if the
documentation is unclear.

## Publication
This package was used and characterized in the publication "XYZ"

## License
This project is licensed under GNU General Public License v3.0.