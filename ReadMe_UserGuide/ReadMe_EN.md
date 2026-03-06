# GAN_ITSC_2026

WGAN-GP based highway cut-in scenario generation for actively exploring demanding test cases.

---

## Project Structure

```
GAN_ITSC_2026/
├── ReadMe_UserGuide/      # Documentation
│   ├── ReadMe_CN.md       # Chinese project overview
│   ├── ReadMe_EN.md       # This file: English project overview
│   ├── UserGuide_CN.md    # Chinese user guide (detailed operations)
│   └── UserGuide_EN.md    # English user guide (detailed operations)
├── setup_env.bat          # One-click environment setup script
├── environment.yml        # Conda environment definition (optional)
├── process_data.py        # Preprocess + 5s normalization (project root)
├── critical_matrix_for_trainingdata.py  # Criticality computation (UN R157, project root)
├── data/
│   ├── original_data/     # Raw scenario CSVs
│   ├── training_data/     # Preprocessed 5s trajectories
│   └── critiality_matrix/ # Criticality output (trainingdata_critiality.csv)
├── logs/
│   └── scenario_process/  # process_data run logs
├── scenario_process/      # Other data processing scripts
│   ├── check_duration.py  # Duration statistics
│   ├── duration_result.txt
├── wgan/                  # WGAN-GP (to be created)
│   ├── dataset.py         # Dataset
│   ├── train.py           # Training
│   ├── generate.py        # Generation
│   └── outputs/           # Model checkpoints and generated samples
└── old/                   # Legacy scripts (reference only)
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| **setup_env.bat** | One-click setup: create conda env `gan_itsc`, install PyTorch (GPU), numpy, pandas, matplotlib |
| **environment.yml** | Conda env config for `conda env create -f`; PyTorch must be installed separately |
| **process_data.py** | Preprocess raw CSVs in `data/original_data/` (translate, flip, 5s normalize), output to `data/training_data/` |
| **critical_matrix_for_trainingdata.py** | Compute per-case criticality (UN R157, THW+TTC), output to `data/critiality_matrix/trainingdata_critiality.csv` |
| **scenario_process/check_duration.py** | Compute duration statistics of cases in `data_after_process/`, output to `duration_result.txt` |
| **scenario_process/duration_result.txt** | Duration stats: total count, 4-5.5s count, most common duration, etc. |
| **ReadMe_UserGuide/ReadMe_CN.md** | Chinese project overview and file list |
| **ReadMe_UserGuide/ReadMe_EN.md** | English project overview and file list |
| **ReadMe_UserGuide/UserGuide_CN.md** | Chinese detailed operation guide |
| **ReadMe_UserGuide/UserGuide_EN.md** | English detailed operation guide |

---

## Data Format

**Input (data/original_data/)**: CSV with columns `frame`, `ego_x`, `ego_y`, `target_x`, `target_y`.

**Output (data/training_data/*_formal.csv)**: `t`, `x_ego`, `y_ego`, `x_tgt`, `y_tgt`, 126 rows (0–5s), coordinates relative to ego start.

**Criticality output (data/critiality_matrix/trainingdata_critiality.csv)**: columns `case_file`, `criticality`; C ∈ [0, 1]; some cases may be blank (no valid conflict-zone timestep). Paper subsection outline: `ReadMe_UserGuide/ITSC_Criticality_Section_Outline.md`.

---

## Quick Start

1. Run `setup_env.bat` to configure environment
2. `conda activate gan_itsc`
3. See `ReadMe_UserGuide/UserGuide_EN.md` for detailed steps
