# Example - Neural Architecture Search with EpiDeNet Architecture
This example demonstrates the implementation of an AbstractModelStudy to optimize the 
EpiDeNet architecture for epileptic seizure detection.

- basemodelstudy.py: implementation of the model study, missing informations about running with CPU or GPU.
- cpu_modelstudy.py: CPU implementation of the model study.
- gpu_modelstudy.py: GPU implementation of the model study.

## Requirements
- Pre-processed CHB-MIT dataset, see examples/pre_process_chb_mit/

## Usage

### GPU

Not yet implemented.

### CPU

```
python cpu_modelstudy.py self-test
python cpu_modelstudy.py setup
./cpu_modelstudy/run_model_study.sh
optuna-dashboard sqlite:///cpu_modelstudy/study_storage.db
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
