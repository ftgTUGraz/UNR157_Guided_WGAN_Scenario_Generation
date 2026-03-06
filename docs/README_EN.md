# UNR157-Based-WGAN

Cut-in scenario generation using WGAN-GP and Risk guidance, aligned with UN R157 criticality evaluation for automated driving test scenarios.

---

## 1. Overview

This project uses **Wasserstein GAN with Gradient Penalty (WGAN-GP)** and a differentiable **Risk** module to generate cut-in driving scenario trajectories under physics and cut-in priors. It supports UN R157 criticality computation and comparison for automated driving system testing and scenario augmentation.

### 1.1 Technical Highlights

| Component | Description |
|-----------|-------------|
| WGAN-GP | Stable training, reduced mode collapse vs. vanilla GAN |
| Risk guidance | 2D longitudinal/lateral proximity as differentiable criticality |
| Physics constraints | Acceleration limits, no reverse, cut-in prior (initial lateral ~4 m, final merge) |
| Collision penalty | Soft-boundary overlap penalty for plausible trajectories |
| UN R157 | THW + TTC criticality, regulation-compliant |

### 1.2 Use Cases

- L2/L3 automated driving system test scenario generation
- Cut-in scenario data augmentation
- Criticality distribution analysis

---

## 2. Project Structure

```
UNR157_Based_WGAN/
├── src/
│   ├── config_mve.py                    # Global configuration
│   ├── dataset_mve.py                   # Training dataset
│   ├── models_mve.py                    # G / D / Risk models
│   ├── process_data_mve.py              # Data preprocessing
│   ├── train_mve.py                     # Training script
│   ├── generate_mve.py                  # Scenario generation
│   ├── Critical_calculate_original_mve.py   # Criticality for original data
│   ├── Critical_calculate_generated_mve.py  # Criticality for generated data
│   └── compare_criticality_mve.py       # Criticality comparison
├── data/
│   ├── original_data/                   # Raw trajectory CSVs
│   ├── original_data_coordinate/        # Standardized coords (*_real.csv)
│   ├── training_data/                   # Training data (*_train.csv)
│   ├── generated_data/                  # Generated scenarios (gen_mve_*.csv)
│   └── critiality_matrix/               # Criticality outputs
├── outputs_mve/checkpoints_mve/         # Training checkpoints
├── logs/                                # Logs
├── docs/                                # Documentation
├── requirements_mve.txt
├── README_CN.md
└── README_EN.md
```

---

## 3. Environment & Installation

- Python 3.8+
- CUDA (optional, for GPU training)

```bash
pip install -r requirements_mve.txt
```

Dependencies: `torch>=1.9.0`, `numpy>=1.20.0`, `pandas>=1.3.0`, `matplotlib>=3.4.0`.

---

## 4. Data Format

### 4.1 Raw Input (`data/original_data/*.csv`)

| Column | Description |
|--------|-------------|
| frame | Frame index |
| ego_x, ego_y | Ego longitudinal/lateral position (m) |
| target_x, target_y | Target longitudinal/lateral position (m) |

### 4.2 Training Data (`*_train.csv`)

| Column | Description |
|--------|-------------|
| t | Time (s) |
| x_ego, y_ego | Ego trajectory (m), y_ego = 0 |
| x_tgt, y_tgt | Target trajectory (m), y_tgt ∈ [0, 4.5] |

5 s, 126 frames, 25 FPS.

---

## 5. Usage

> Run all commands from the project root.

### 5.1 Data Preprocessing

Place raw CSVs in `data/original_data/`, then:

```bash
python -m src.process_data_mve
```

- Step1: Coordinate standardization → `data/original_data_coordinate/*_real.csv`
- Step2: Normalize to 5 s, 126 frames → `data/training_data/*_train.csv`

### 5.2 Training

```bash
python -m src.train_mve
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | data/training_data | Training data directory |
| `--out_dir` | outputs_mve/checkpoints_mve | Checkpoint directory |
| `--epochs` | 500 | Number of epochs |
| `--batch_size` | 64 | Batch size |
| `--device` | cuda | cuda / cpu |
| `--save_every` | 100 | Save interval (epochs) |
| `--max_samples` | - | Max samples (debug) |

### 5.3 Generate Scenarios

```bash
python -m src.generate_mve
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | checkpoint_mve_final.pt | Checkpoint path |
| `--out_dir` | data/generated_data | Output directory |
| `--n_samples` | 100 | Number of samples |

### 5.4 Criticality Computation

```bash
# Original data
python -m src.Critical_calculate_original_mve

# Generated data
python -m src.Critical_calculate_generated_mve
```

Results in `data/critiality_matrix/`.

### 5.5 Criticality Comparison

```bash
python -m src.compare_criticality_mve
```

Reports mean criticality (valid cases) and ratio of criticality > 0.8.

---

## 6. Configuration

Main config: `src/config_mve.py` — paths, FPS, frame count, model structure, training params, Risk and constraint weights.

---

## 7. References

- UN R157: Type approval for L2/L3 automated driving systems
- WGAN-GP: Improved Training of Wasserstein GANs

---

## 8. License

See LICENSE in the project root.
