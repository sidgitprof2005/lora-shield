# LoRA Shield 🛡️

LoRA Shield is a machine learning security framework designed to detect backdoor and poisoning attacks inside parameter-efficient LoRA adapters. The framework extracts Singular Value Decomposition (SVD) spectral properties and weight distribution metrics from low-rank matrices to identify anomalous weights injected during poisoned fine-tuning epochs.

## Architecture

```
     [Uploaded LoRA Adapter (.safetensors / .bin)]
                          │
                          ▼
              ┌──────────────────────┐
              │ LoRAFeatureExtractor │
              └──────────┬───────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
  [Spectral Features (SVD)]    [Statistical Features]
  - Mean, Std, Max             - Weight Mean & Std
  - Concentration (S[0]/sum)   - Max Absolute Weight
  - Spectral Entropy           - Kurtosis & Skewness
           └─────────────┬─────────────┘
                         │ (Averaged 10D Vector)
                         ▼
               ┌──────────────────┐
               │ XGBoost Detector │
               └─────────┬────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
   [ Verdict: CLEAN ]          [ Verdict: POISONED ]
   - Safe to Deploy            - Risk Confidence %
   - Normal Distribution       - List Suspicious Layers
```

---

## Features

- **Double Feature Set**: Extracts SVD-based singular features (entropy, concentration) combined with statistical shape measurements (kurtosis, skewness, variance).
- **In-Memory Scanning**: Streamlit interface processes `.safetensors` and `.bin` uploads directly in RAM.
- **Layer-Level Auditing**: Inspects and highlights the exact layers showing backdoored patterns instead of just declaring a global warning.
- **Plotly Visuals**: Generates interactive charts representing SVD profiles, parameter histograms, and feature importances.
- **Automatic Setup**: Auto-generates classifier datasets and model parameters on execution if missing.

---

## Installation

1. Clone or copy this repository to your local workspace:
   ```bash
   cd lora_shield/
   ```

2. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Run the Orchestrator Pipeline
To download clean adapters, train poisoned adapters, run the extractor, train the XGBoost classifier, and write performance results:
```bash
python main.py
```
*(Use `--max_samples 1000 --epochs 1` to run a fast dry-run for test verification).*

### 2. Start the Streamlit Dashboard UI
```bash
streamlit run app.py
```

### 3. Run Unit Tests
To execute the test suite covering feature extraction and model prediction:
```bash
python -m unittest discover tests
```

---

## Example Outputs & UI Previews

### Safe Adapter (Clean)
*Placeholder: Normal Gaussian bell curve representation of weights, Green Success Alert `✅ SAFE TO DEPLOY`.*

### Backdoored Adapter (Poisoned)
*Placeholder: Spike in SVD concentration values, Red Error Alert `☠️ BACKDOOR DETECTED`, Bar chart highlighting specific suspicious layers.*

---

## Research Paper Reference

The architecture and feature selections in this framework are based on the methodology outlined in the following paper:
> **FlipBoost: Spectral Entropy and High-Order Statistical Defense against Low-Rank Backdoor Attacks** (2025)
