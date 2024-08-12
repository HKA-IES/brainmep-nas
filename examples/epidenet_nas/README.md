# Example - Neural Architecture Search with EpiDeNet Architecture
This example demonstrates the implementation of an AbstractModelStudy to optimize the 
EpiDeNet architecture for epileptic seizure detection.

- basemodelstudy.py: implementation of the model study, missing informations about running with CPU or GPU.
- cpu_modelstudy.py: CPU implementation of the model study.
- gpu_modelstudy.py: GPU implementation of the model study.

## Requirements
- Pre-processed CHB-MIT dataset, see examples/pre_process_chb_mit/
- [optuna-dashboard](https://github.com/optuna/optuna-dashboard)

## Usage

First, set DATASET_DIR in basemodelstudy.py to the directory containing the
pre-processed CHB-MIT data (see examples/pre_process_chb_mit/)

### GPU

```
python gpu_modelstudy.py self-test
python gpu_modelstudy.py setup-inner-loops
./gpu_modelstudy/run_all_inner_loops.sh
optuna-dashboard sqlite:///gpu_modelstudy/study_storage.db
python gpu_modelstudy.py setup-outer-loop
./gpu_modelstudy/run_outer_loop.sh
```

### CPU

```
python cpu_modelstudy.py self-test
python cpu_modelstudy.py setup-inner-loops
./cpu_modelstudy/run_all_inner_loops.sh
optuna-dashboard sqlite:///cpu_modelstudy/study_storage.db
python cpu_modelstudy.py setup-outer-loop
./cpu_modelstudy/run_outer_loop.sh
```

## Detailed description
- Dataset: CHB-MIT for patient 5 with leave-one-out nested cross-validation.
- Search space: EpiDeNet architecture[1] where, for each convolutional layer, the number of kernels and their size is left free.
- Search strategy: MOTPE
- Performance estimation strategy: performance of a trial is the average performance of the inner folds.
  - Each inner fold is trained for 50 epochs with early stopping on the validation loss with a patience of 5.
  - Objectives: maximize sample_sensitivity (see AccuracyMetrics) and minimize energy (see MltkHardwareMetrics)

References: 
[1] T. M. Ingolfsson et al., “EpiDeNet: An Energy-Efficient 
Approach to Seizure Detection for Embedded Systems.” arXiv, Aug. 28, 2023. 
Accessed: Sep. 20, 2023. [Online]. Available: http://arxiv.org/abs/2309.07135
