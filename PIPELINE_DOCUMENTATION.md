# Data Cleaning Pipeline Documentation
## Airbnb Price Prediction Project

---

## 🎯 Overview

This document describes the **function-based data cleaning pipeline** with **calendar aggregation** and **proper train/val/test splitting** to prevent data leakage.

---

## 📋 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Load Data                                           │
│  • listings_details.csv                                     │
│  • calendar.csv                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Calendar Aggregation                                │
│  • Clean calendar price & availability                      │
│  • Group by listing_id                                      │
│  • Create features:                                         │
│    - avg_calendar_price                                     │
│    - min_calendar_price                                     │
│    - max_calendar_price                                     │
│    - availability_rate                                      │
│    - calendar_days_count                                    │
│    - calendar_available_days                                │
│  • Merge with listings                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Train/Val/Test Split (60/20/20)                     │
│  ⚠️ CRITICAL: Split BEFORE any transformations!             │
│  • Remove duplicates                                        │
│  • Remove missing/invalid prices                            │
│  • Split: 60% train, 20% val, 20% test                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Define Preprocessing Function                       │
│  preprocess_data(X_train, X_val, X_test, y_train)          │
│                                                             │
│  1. Drop irrelevant columns                                │
│  2. Type conversion (price, percentage, boolean, numeric)  │
│  3. Logic error fixing (min > max nights)                  │
│  4. Drop high missing columns (>70%)                       │
│  5. Domain knowledge fills                                 │
│  6. Feature engineering (dates, text, amenities)           │
│  7. Cleanup (drop original text/date columns)              │
│  8. Categorical encoding (one-hot + target encoding)       │
│  9. Handle remaining missing values (TRAIN median)         │
│  10. Outlier treatment (TRAIN quantiles)                   │
│                                                             │
│  ✅ All transformations FIT on TRAIN, TRANSFORM all splits │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Apply Preprocessing                                 │
│  X_train_clean, X_val_clean, X_test_clean, features, enc   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Scaling (3 scalers available)                       │
│  • StandardScaler (default, best for most models)          │
│  • MinMaxScaler (neural networks)                          │
│  • RobustScaler (outlier-resistant)                        │
│                                                             │
│  All FIT on TRAIN, TRANSFORM all splits                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Save Processed Data                                 │
│  • Unscaled CSV files                                       │
│  • Scaled NumPy arrays (3 versions)                        │
│  • Feature names                                            │
│  • Scalers & encoders (pickle)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Features

### ✅ **No Data Leakage**
- Data split **BEFORE** any transformations
- All statistics (mean, median, quantiles) computed from **training data only**
- Transformations applied consistently across train/val/test

### ✅ **Calendar Integration**
- Aggregates `calendar.csv` at the listing level
- Creates 6 new features capturing pricing and availability patterns
- Left join preserves all listings (handles missing calendar data)

### ✅ **Function-Based Architecture**
- Single `preprocess_data()` function for all transformations
- Reproducible and maintainable
- Easy to modify or extend
- Returns all processed datasets and metadata

### ✅ **Comprehensive Feature Engineering**
- **Date features**: Host tenure, review recency, cyclical encoding
- **Text features**: Length and word count for all text fields
- **Amenities**: Binary flags for key amenities
- **Calendar**: Pricing statistics and availability metrics

### ✅ **Proper Split Ratio**
- **60% Train**: Maximum data for model training
- **20% Validation**: Hyperparameter tuning and model selection
- **20% Test**: Final unbiased performance evaluation

---

## 📊 Data Flow

### Input Files
```
main_dataset/
├── listings_details.csv    (20,030 listings × 96 features)
└── calendar.csv            (7,310,950 rows × 4 features)
```

### Output Files
```
processed_data/
├── train_unscaled.csv           # Unscaled training data
├── val_unscaled.csv             # Unscaled validation data
├── test_unscaled.csv            # Unscaled test data
│
├── X_train_standard.npy         # StandardScaler
├── X_val_standard.npy
├── X_test_standard.npy
│
├── X_train_minmax.npy           # MinMaxScaler
├── X_val_minmax.npy
├── X_test_minmax.npy
│
├── X_train_robust.npy           # RobustScaler
├── X_val_robust.npy
├── X_test_robust.npy
│
├── y_train.npy                  # Target values
├── y_val.npy
├── y_test.npy
│
├── feature_names.csv            # List of all features
├── scaler_standard.pkl          # Fitted StandardScaler
├── scaler_minmax.pkl            # Fitted MinMaxScaler
├── scaler_robust.pkl            # Fitted RobustScaler
└── encoders.pkl                 # All fitted encoders/stats
```

---

## 🛠️ Usage

### Running the Pipeline
```python
# Simply run all cells in data_cleaning.ipynb
# The notebook will:
# 1. Load data
# 2. Aggregate calendar
# 3. Split data
# 4. Define preprocessing function
# 5. Apply preprocessing
# 6. Scale data
# 7. Save all outputs
```

### Loading Processed Data for Modeling

**Option 1: Unscaled Data (CSV)**
```python
import pandas as pd

train_df = pd.read_csv('processed_data/train_unscaled.csv')
val_df = pd.read_csv('processed_data/val_unscaled.csv')
test_df = pd.read_csv('processed_data/test_unscaled.csv')

X_train = train_df.drop(columns=['price'])
y_train = train_df['price']
```

**Option 2: Scaled Data (NumPy)**
```python
import numpy as np

# StandardScaler (recommended for most models)
X_train = np.load('processed_data/X_train_standard.npy')
X_val = np.load('processed_data/X_val_standard.npy')
X_test = np.load('processed_data/X_test_standard.npy')

y_train = np.load('processed_data/y_train.npy')
y_val = np.load('processed_data/y_val.npy')
y_test = np.load('processed_data/y_test.npy')
```

**Option 3: Load Scalers for New Data**
```python
import pickle

# Load fitted scaler
with open('processed_data/scaler_standard.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Transform new data
X_new_scaled = scaler.transform(X_new)
```

---

## 📈 Feature Categories

### Calendar-Derived Features (6)
- `avg_calendar_price`: Mean price from calendar
- `min_calendar_price`: Minimum price from calendar
- `max_calendar_price`: Maximum price from calendar
- `availability_rate`: Proportion of available days
- `calendar_days_count`: Total days in calendar
- `calendar_available_days`: Number of available days

### Date Features (~10)
- Host tenure (days, years)
- Host since (year, month, day of week)
- Cyclical encoding (sin/cos for month)
- Days since first/last review
- Review period length

### Text Features (~20)
- Length and word count for:
  - name, summary, space, description
  - neighborhood_overview, notes, transit
  - access, interaction, house_rules

### Amenity Features (7)
- `amenities_count`: Total number of amenities
- Binary flags: wifi, kitchen, tv, parking, ac, heating

### Original Listing Features (~100+)
- Property details (bedrooms, bathrooms, accommodates)
- Location (latitude, longitude, neighborhood)
- Pricing (cleaning_fee, security_deposit)
- Reviews (scores, counts)
- Host information (response rate, superhost status)
- Availability (30, 60, 90, 365 days)

---

## ⚠️ Important Notes

### Data Leakage Prevention
1. **Split first**: Always split before any transformations
2. **Fit on train**: All statistics computed from training data
3. **Transform all**: Apply learned transformations to val/test
4. **Never use val/test**: Don't peek at validation/test during preprocessing

### Missing Values Strategy
| Type | Strategy |
|------|----------|
| High missing (>70%) | Drop column |
| Security deposit | Fill with 0 |
| Review scores | Fill with 0 |
| Text fields | Fill with 'Unknown' |
| Categorical | Fill with training mode |
| Numeric | Fill with training median |

### Outlier Treatment
| Feature | Method |
|---------|--------|
| minimum_nights | Cap at 365 |
| maximum_nights | Cap at 730 |
| accommodates | Cap at 16 |
| cleaning_fee | Winsorize at 99th percentile (train) |
| security_deposit | Winsorize at 99th percentile (train) |

### Encoding Strategy
| Cardinality | Method |
|-------------|--------|
| < 10 unique values | One-hot encoding |
| ≥ 10 unique values | Target encoding (fit on train) |
| Boolean | Convert to 0/1 |

---

## 🔄 Modifying the Pipeline

### Adding New Features
Add feature engineering code inside the `preprocess_data()` function in Step 6 (Feature Engineering section).

### Changing Split Ratio
Modify the `train_test_split` parameters in Step 3:
```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42  # Adjust test_size
)
```

### Adding Another Scaler
Add a new scaler in Step 6:
```python
scaler_new = YourScaler()
X_train_new = scaler_new.fit_transform(X_train_clean)
X_val_new = scaler_new.transform(X_val_clean)
X_test_new = scaler_new.transform(X_test_clean)
```

---

## ✅ Quality Checks

The pipeline includes automatic verification:
- ✅ No NaN values after preprocessing
- ✅ No NaN values after scaling
- ✅ Consistent shapes across train/val/test
- ✅ Proper scaling statistics
- ✅ Feature alignment across splits

---

## 📞 Questions?

If you encounter issues or need modifications:
1. Check the cell outputs for error messages
2. Verify input file paths are correct
3. Ensure all required libraries are installed
4. Review the `encoders` dictionary for transformation details

---

**Last Updated**: November 2025  
**Pipeline Version**: 2.0 (Function-based with Calendar Integration)

