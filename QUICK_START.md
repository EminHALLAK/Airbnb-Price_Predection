# Quick Start Guide
## Airbnb Price Prediction - Data Cleaning Pipeline

---

## 🚀 Getting Started (3 Steps)

### Step 1: Run the Notebook
```bash
# Open Jupyter notebook
jupyter notebook data_cleaning.ipynb

# OR run all cells programmatically
jupyter nbconvert --to notebook --execute data_cleaning.ipynb
```

### Step 2: Check Output
Look for the `processed_data/` directory with all output files.

### Step 3: Load Data for Modeling
```python
import numpy as np

# Load scaled data (ready for ML)
X_train = np.load('processed_data/X_train_standard.npy')
y_train = np.load('processed_data/y_train.npy')

X_val = np.load('processed_data/X_val_standard.npy')
y_val = np.load('processed_data/y_val.npy')

# Train your model!
```

---

## 📊 What You Get

### Data Splits
- **Train**: 60% (~12,000 samples)
- **Validation**: 20% (~4,000 samples)
- **Test**: 20% (~4,000 samples)

### Features
- **~150-200 features** (depends on one-hot encoding)
- Includes **6 calendar-derived features**:
  - avg_calendar_price
  - min_calendar_price
  - max_calendar_price
  - availability_rate
  - calendar_days_count
  - calendar_available_days

### File Formats
- **CSV**: Unscaled data with column names
- **NumPy**: Scaled data ready for ML models
- **Pickle**: Fitted scalers and encoders

---

## 🔧 Common Tasks

### Use Different Scaler
```python
# MinMaxScaler (good for neural networks)
X_train = np.load('processed_data/X_train_minmax.npy')

# RobustScaler (good for outliers)
X_train = np.load('processed_data/X_train_robust.npy')

# StandardScaler (default, good for most models)
X_train = np.load('processed_data/X_train_standard.npy')
```

### Get Feature Names
```python
import pandas as pd

features = pd.read_csv('processed_data/feature_names.csv')
print(features['feature'].tolist())
```

### Transform New Data
```python
import pickle

# Load fitted scaler
with open('processed_data/scaler_standard.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load encoders (for reference)
with open('processed_data/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Transform new data
X_new_scaled = scaler.transform(X_new_preprocessed)
```

---

## ⚡ Pipeline Summary

```
Load Data → Aggregate Calendar → Split (60/20/20) → Preprocess → Scale → Save
```

**Total Execution Time**: ~2-5 minutes (depending on hardware)

---

## ✅ Quality Guarantees

- ✅ **No data leakage**: All transformations fit on training data only
- ✅ **No NaN values**: Complete imputation with proper handling
- ✅ **Consistent features**: All splits have identical columns
- ✅ **Proper scaling**: Train has mean≈0, std≈1 (StandardScaler)
- ✅ **Calendar integrated**: 6 new features from calendar data

---

## 📁 File Structure

```
project/
├── data_cleaning.ipynb              # Main pipeline notebook
├── PIPELINE_DOCUMENTATION.md         # Detailed documentation
├── QUICK_START.md                    # This file
│
├── main_dataset/
│   ├── listings_details.csv         # Input: Listings data
│   └── calendar.csv                  # Input: Calendar data
│
└── processed_data/                   # Output: All processed files
    ├── train_unscaled.csv            # Unscaled training data
    ├── val_unscaled.csv              # Unscaled validation data
    ├── test_unscaled.csv             # Unscaled test data
    │
    ├── X_train_standard.npy          # Scaled features (StandardScaler)
    ├── X_val_standard.npy
    ├── X_test_standard.npy
    │
    ├── X_train_minmax.npy            # Scaled features (MinMaxScaler)
    ├── X_val_minmax.npy
    ├── X_test_minmax.npy
    │
    ├── X_train_robust.npy            # Scaled features (RobustScaler)
    ├── X_val_robust.npy
    ├── X_test_robust.npy
    │
    ├── y_train.npy                   # Target values
    ├── y_val.npy
    ├── y_test.npy
    │
    ├── feature_names.csv             # Feature list
    ├── scaler_standard.pkl           # Fitted scalers
    ├── scaler_minmax.pkl
    ├── scaler_robust.pkl
    └── encoders.pkl                  # Fitted encoders
```

---

## 🎯 Next Steps

1. **Explore the data**: Check distributions, correlations
2. **Baseline model**: Start with Linear Regression
3. **Advanced models**: Try Random Forest, XGBoost, Neural Networks
4. **Hyperparameter tuning**: Use validation set
5. **Final evaluation**: Use test set only once

---

## 💡 Tips

- Use **StandardScaler** for most traditional ML models (Linear, SVM, KNN)
- Use **MinMaxScaler** for neural networks and algorithms sensitive to feature ranges
- Use **RobustScaler** if your data still has outliers after preprocessing
- Check `feature_names.csv` to understand what each column represents
- The `encoders.pkl` file contains all fitted transformations for reference

---

## ⚠️ Important Notes

1. **Never retrain on validation/test data**: Only use for evaluation
2. **Don't look at test metrics during development**: Wait until final evaluation
3. **Split is deterministic** (`random_state=42`): Rerunning gives same splits
4. **Calendar features may have NaN**: Listings without calendar data get NaN → filled with median

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check `main_dataset/` directory exists |
| Memory error | Close other programs, process in chunks |
| Different shapes | Ensure same features across splits (check alignment) |
| NaN in output | Check final summary, should be 0 NaN values |
| Encoding error (Windows) | UTF-8 encoding set at start of notebook |

---

## 📚 Learn More

- **Detailed docs**: See `PIPELINE_DOCUMENTATION.md`
- **Workflow**: See `data_cleaning.ipynb` with inline comments
- **Previous work**: Check `FINAL_SUMMARY.md` and `DATA_LEAKAGE_ANALYSIS_AND_FIXES.md`

---

**Happy Modeling! 🎉**

