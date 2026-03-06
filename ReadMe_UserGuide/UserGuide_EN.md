# GAN_ITSC_2026 User Guide

This document describes step-by-step operations for each feature. New features will be added here as they are implemented.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Duration Statistics](#3-duration-statistics)
4. [Criticality Computation](#4-criticality-computation)
5. [WGAN-GP Training](#5-wgan-gp-training) (to be added)
6. [Trajectory Generation](#6-trajectory-generation) (to be added)
7. [Trajectory Visualization](#7-trajectory-visualization) (to be added)

---

## 1. Environment Setup

### Description

Creates the Anaconda environment `gan_itsc` and installs PyTorch (GPU), numpy, pandas, and matplotlib for training, generation, and plotting.

### Prerequisites

- Anaconda or Miniconda installed
- For GPU: NVIDIA GPU with appropriate drivers

### Steps

1. Open **Anaconda Prompt** (do not use PowerShell)
2. Change to project directory:
   ```cmd
   cd /d c:\GAN_ITSC_2026
   ```
3. Run setup script:
   ```cmd
   setup_env.bat
   ```
4. Wait for completion (about 3–5 minutes)
5. Verify: output should show `CUDA available: True` or `False` (CPU mode)

### Daily Usage

Activate the environment before use:
```cmd
conda activate gan_itsc
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| conda not found | Install Anaconda and use its Anaconda Prompt |
| PyTorch DLL error | Script falls back to CPU; or manually `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| CUDA not available | Check drivers; or use CPU PyTorch |

---

## 2. Data Preprocessing (process_data.py)

### Description

Preprocesses raw cut-in scenario CSVs in `data/original_data/` and writes results to `data/training_data/`. Processing includes:

- Translate coordinates so ego start is origin
- Flip x if ego moves along -x so ego uniformly moves in +x
- Take absolute value of y_tgt
- Time: t = (frame - first_frame) / 25, starting from 0
- Normalize to 5 seconds: linear interpolation resample to 126 frames (5s @ 25Hz)

### Input / Output

| Item | Path / Note |
|------|-------------|
| Input dir | `C:\GAN_ITSC_2026\data\original_data` |
| Output dir | `C:\GAN_ITSC_2026\data\training_data` |
| Input format | CSV with columns: `frame`, `ego_x`, `ego_y`, `target_x`, `target_y` |
| Output format | Each row: `t`, `x_ego`, `y_ego`, `x_tgt`, `y_tgt`, 126 rows (0–5s) |
| Log dir | `logs/scenario_process/`, filename `process_data_YYYYMMDD_HHMMSS.log` |

### Steps

1. Activate environment: `conda activate gan_itsc`
2. Change to project directory: `cd /d c:\GAN_ITSC_2026`
3. Run:
   ```cmd
   python process_data.py
   ```
4. Check logs: each run creates a log file under `logs/scenario_process/`

### Logging

- Console: INFO level, progress and summary
- Log file: DEBUG level, per-file status, errors
- Format: `YYYY-MM-DD HH:MM:SS | LEVEL | message`

---

## 3. Duration Statistics (check_duration.py)

### Description

Computes duration statistics for cases in the target directory and counts cases in the 4–5.5 second range. Note: after process_data, all cases are normalized to 5s; this script is mainly for raw data or legacy output.

### Steps

1. Activate environment: `conda activate gan_itsc`
2. Change to project directory: `cd /d c:\GAN_ITSC_2026`
3. Run:
   ```cmd
   python scenario_process/check_duration.py
   ```
4. Results written to `scenario_process/duration_result.txt`. For training_data (normalized to 5s), all durations will be 5.0s.

### Output Contents

- Total case count
- Count of cases in 4–5.5 s
- Most common duration and its count
- Top 10 duration distribution
- Min and max duration

---

## 4. Criticality Computation (critical_matrix_for_trainingdata.py)

### Description

Computes **criticality** per cut-in scenario following UN R157, for filtering or weighting high-criticality samples. The metric combines **time headway (THW)** and **time-to-collision (TTC)** and is evaluated only on timesteps where the target is ahead and within 2.75 m laterally; the maximum over time is taken as the scenario criticality. Criticality C ∈ [0, 1], with C = 1 corresponding to the theoretically most critical situation (very small headway or imminent collision).

### Input / Output

| Item | Path / Note |
|------|-------------|
| Input dir | `C:\GAN_ITSC_2026\data\original_data` (same raw CSVs as process_data) |
| Output dir | `C:\GAN_ITSC_2026\data\critiality_matrix` |
| Output file | `trainingdata_critiality.csv`, columns: `case_file`, `criticality` |
| Input format | CSV with columns: `frame`, `ego_x`, `ego_y`, `target_x`, `target_y` |

### Steps

1. Activate environment: `conda activate gan_itsc`
2. Change to project root: `cd /d c:\GAN_ITSC_2026`
3. Run:
   ```cmd
   python critical_matrix_for_trainingdata.py
   ```
4. Console prints the output path, e.g. `Criticality CSV written: C:\GAN_ITSC_2026\data\critiality_matrix\trainingdata_critiality.csv`

### Notes

- **Blank criticality**: If a case never has a timestep where the target is ahead and within 2.75 m laterally, its criticality is left blank (NaN) in the CSV. Such cases can be treated as having no valid conflict zone and excluded from criticality-based ranking.
- **Range**: Valid criticality is in [0, 1]; 1 is the theoretical maximum.
- **Relation to process_data**: This script uses raw frame length (no 5s normalization); alignment (translate, flip x, \|y_tgt\|) matches process_data. Paper subsection outline: `ReadMe_UserGuide/ITSC_Criticality_Section_Outline.md`.

---

## 5. WGAN-GP Training

*(To be implemented; placeholder)*

### Description

Train generator and discriminator with WGAN-GP to learn cut-in trajectory distribution; optionally add harshness guidance.

### Steps

1. Activate environment
2. Change to `wgan/` directory
3. Run: `python train.py` or `python train_gpu_only.py`

### Output

- Model saved to `wgan/outputs/checkpoint.pt`
- Sample generations saved every 100 epochs

---

## 6. Trajectory Generation

*(To be implemented; placeholder)*

### Description

Load trained model and generate new cut-in trajectories.

### Steps

1. Ensure `checkpoint.pt` exists
2. Run: `python generate.py -n 20`
3. Results saved to `outputs/generated/`

---

## 7. Trajectory Visualization

*(To be implemented; placeholder)*

### Description

Plot ego and target trajectories in top-down view (x-y).

### Steps

1. Single trajectory: `python plot_trajectory.py path/to/xxx.csv`
2. All generated trajectories: `python plot_all_generated.py`

---

## Changelog

- Initial: environment setup, data preprocessing, duration statistics
- Added: criticality computation (critical_matrix_for_trainingdata.py) description and steps
