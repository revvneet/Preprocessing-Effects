# Quick Start Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## Running Experiments

1. **Baseline Model**
   ```bash
   cd experiments/baseline
   python TrainModel.py
   ```

2. **Missing Values - KNN Imputation**
   ```bash
   cd experiments/missing_values_imputed
   python TrainModel.py
   ```

3. **Missing Values - Drop Rows**
   ```bash
   cd experiments/missing_values_dropped
   python TrainModel.py
   ```

4. **SMOTE + Scaling**
   ```bash
   cd experiments/smote_scaling
   python TrainModel.py
   ```

5. **Max Iter Increase**
   ```bash
   cd experiments/max_iter
   python TrainModel.py
   ```

6. **Recall Fix (Class Weight Balanced)**
   ```bash
   cd experiments/recall_fix
   python TrainModel.py
   ```

## Output Files

Each experiment creates:
- `trained_model.pkl` - Saved model
- Classification report printed to console

PNG visualizations are stored in `../outputs/<variant>/`