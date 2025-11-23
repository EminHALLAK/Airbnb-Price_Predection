# Airbnb Price Predection

# Data Cleaning Documentation

## Overview
This document describes the comprehensive data cleaning and preprocessing pipeline for the Airbnb price prediction project.

## Dataset Information

### Input Files
- `listings_details.csv` - Detailed listing information (20,030 listings, 96 columns)
- `calendar.csv` - Availability and pricing calendar data
- `reviews.csv` - Guest reviews
- `neighbourhoods.csv` - Amsterdam neighborhood information

### Target Variable
- **price** - Nightly price in local currency (EUR)

## Data Cleaning Pipeline

### Step 1: Initial Data Cleaning
✅ **Duplicate Removal**
- Removed duplicate records based on listing ID
- Verified no duplicate IDs remain

✅ **Column Dropping**
- Removed URLs, image links (thumbnail_url, picture_url, etc.)
- Removed redundant IDs (host_id, scrape_id)
- Removed columns with 100% missing values
- Removed non-predictive text descriptions

✅ **Type Conversion**
- **Price columns**: Converted from string ($X,XXX.XX) to float
  - price, weekly_price, monthly_price, security_deposit, cleaning_fee, extra_people
- **Percentage columns**: Converted from string (XX%) to decimal (0.XX)
  - host_response_rate, host_acceptance_rate
- **Boolean columns**: Converted from 't'/'f' to True/False
  - host_is_superhost, instant_bookable, etc.
- **Numeric columns**: Converted object types to numeric
  - accommodates, bathrooms, bedrooms, beds, review scores, etc.
- **Date columns**: Converted to datetime
  - host_since, first_review, last_review

✅ **Logic Error Detection**
- Removed negative prices
- Fixed minimum_nights > maximum_nights conflicts
- Identified unreasonable values

### Step 2: Missing Value Treatment

**Strategy:**
- \>70% missing: Drop column
- 30-70% missing: Domain-specific imputation
- <30% missing: Statistical imputation

✅ **Specific Treatments:**
- **weekly_price**: Filled with `price * 7 * 0.9` (typical discount)
- **monthly_price**: Filled with `price * 30 * 0.8` (typical discount)
- **security_deposit**: Filled with 0 (no deposit required)
- **cleaning_fee**: Filled with median value
- **bathrooms, bedrooms, beds**: Filled with median
- **review scores**: Filled with 0 (no reviews yet)
- **reviews_per_month**: Filled with 0 (no reviews yet)
- **host_neighbourhood**: Filled with neighbourhood_cleansed
- **Categorical columns**: Filled with mode or 'Unknown'
- **Boolean columns**: Filled with False

### Step 3: Date/Time Feature Engineering

✅ **Created Features:**
- **host_tenure_days**: Days since host joined
- **host_tenure_years**: Years since host joined
- **days_since_first_review**: Recency of first review
- **days_since_last_review**: Recency of last review (9999 for no reviews)
- **review_period_days**: Days between first and last review
- **host_since_year**: Year host joined
- **host_since_month**: Month host joined
- **host_since_dayofweek**: Day of week host joined
- **host_since_month_sin**: Sine encoding of month (cyclical)
- **host_since_month_cos**: Cosine encoding of month (cyclical)

### Step 4: Text Feature Processing

✅ **Length Features:**
- Created `_length` and `_word_count` for text columns:
  - name, summary, space, description, neighborhood_overview
  - notes, transit, access, interaction, house_rules

✅ **Amenity Features:**
- **amenities_count**: Total number of amenities
- **has_wifi**: Binary flag
- **has_kitchen**: Binary flag
- **has_tv**: Binary flag
- **has_parking**: Binary flag
- **has_ac**: Binary flag
- **has_heating**: Binary flag

✅ **Host Verification:**
- **host_verifications_count**: Number of verification methods

### Step 5: Outlier Detection and Treatment

✅ **IQR Method:**
- Detected outliers using Interquartile Range (IQR)
- Analyzed outlier percentage for key features

✅ **Treatment Methods:**

**Price (Target Variable):**
- Removed extreme outliers (price = 0 or > 99th percentile × 2)
- Winsorized to 1st-99th percentile range

**Other Features:**
- **minimum_nights**: Capped at 365 days
- **maximum_nights**: Capped at 730 days (2 years)
- **cleaning_fee**: Winsorized at 99th percentile
- **security_deposit**: Winsorized at 99th percentile
- **accommodates**: Capped at 16 people

### Step 6: Categorical Encoding

✅ **Strategy by Cardinality:**

**Low Cardinality (<10 unique values):**
- Method: **One-Hot Encoding**
- Creates binary columns for each category
- Example: room_type → room_type_Entire, room_type_Private, etc.

**Medium Cardinality (10-50 unique values):**
- Method: **Target Encoding**
- Maps each category to mean price
- Smoothed with global mean for rare categories

**High Cardinality (>50 unique values):**
- Method: **Target Encoding**
- More efficient than one-hot for high cardinality
- Example: neighbourhood_cleansed → neighbourhood_cleansed_target_encoded

**Boolean Features:**
- Converted to integer (0/1)

### Step 7: Train/Validation/Test Split

✅ **Split Strategy:**
- **Training set**: 60% (~12,018 samples)
- **Validation set**: 20% (~4,006 samples)
- **Test set**: 20% (~4,006 samples)

✅ **Approach:**
- First split: 80% train+val, 20% test
- Second split: 75% train, 25% val (of the 80%)
- Random state: 42 (reproducibility)

### Step 8: Scaling and Normalization

✅ **Three Scaling Methods Provided:**

**1. StandardScaler (Z-score normalization):**
```python
X_scaled = (X - mean) / std
```
- Mean = 0, Std = 1
- Best for: Normally distributed features, algorithms assuming standard normal distribution
- Use with: Linear Regression, Logistic Regression, Neural Networks, SVM

**2. MinMaxScaler (0-1 normalization):**
```python
X_scaled = (X - min) / (max - min)
```
- Range: [0, 1]
- Best for: Bounded features, algorithms sensitive to feature scale
- Use with: Neural Networks, K-Nearest Neighbors

**3. RobustScaler (robust to outliers):**
```python
X_scaled = (X - median) / IQR
```
- Uses median and IQR instead of mean and std
- Best for: Data with outliers, non-normal distributions
- Use with: Any algorithm when outliers are present

**⚠️ Important:** Scaler is fit ONLY on training data, then applied to validation and test data.

## Output Files

### CSV Files (Human-readable)
- `listings_processed_unscaled.csv` - Full processed dataset with all engineered features
- `train_unscaled.csv` - Training set (unscaled)
- `val_unscaled.csv` - Validation set (unscaled)
- `test_unscaled.csv` - Test set (unscaled)
- `feature_names.csv` - List of all feature names

### NumPy Arrays (ML-ready)
- `X_train_standard.npy`, `X_val_standard.npy`, `X_test_standard.npy` - StandardScaler
- `X_train_minmax.npy`, `X_val_minmax.npy`, `X_test_minmax.npy` - MinMaxScaler
- `X_train_robust.npy`, `X_val_robust.npy`, `X_test_robust.npy` - RobustScaler
- `y_train.npy`, `y_val.npy`, `y_test.npy` - Target values

## Feature Summary

### Numeric Features
- Property characteristics: accommodates, bathrooms, bedrooms, beds
- Pricing: cleaning_fee, security_deposit, extra_people
- Availability: availability_30, availability_60, availability_90, availability_365
- Reviews: number_of_reviews, review_scores_*, reviews_per_month
- Host: host_listings_count, host_total_listings_count, calculated_host_listings_count
- Location: latitude, longitude
- Tenure: host_tenure_days, host_tenure_years
- Recency: days_since_first_review, days_since_last_review, review_period_days
- Text length: *_length, *_word_count features

### Binary Features
- Amenity flags: has_wifi, has_kitchen, has_tv, has_parking, has_ac, has_heating
- Boolean features: host_is_superhost, instant_bookable, is_business_travel_ready, etc.
- One-hot encoded categories

### Encoded Categorical Features
- Target encoded: neighbourhood, property_type, cancellation_policy, etc.
- One-hot encoded: room_type, bed_type, host_response_time, etc.

## Data Quality Metrics

### Before Cleaning
- Total listings: 20,030
- Total features: 96
- Missing values: High (many columns >50% missing)
- Data types: Mixed (many stored as object instead of numeric)

### After Cleaning
- Total listings: ~20,000 (after removing extreme outliers)
- Total features: ~100-150 (after feature engineering and encoding)
- Missing values: 0 (all handled)
- Data types: All numeric (ready for ML)

## Best Practices Implemented

✅ **No Data Leakage:**
- Target encoding computed on training data only
- Scalers fit on training data only
- Test set completely isolated

✅ **Reproducibility:**
- Random state set for splits
- All steps documented
- Feature names saved

✅ **Robustness:**
- Multiple outlier handling methods
- Three scaling options
- Conservative imputation strategies

✅ **Efficiency:**
- Numpy arrays for large datasets
- Dropped unnecessary columns early
- Efficient encoding strategies

## Next Steps for Modeling

### 1. Feature Selection
- Correlation analysis
- Feature importance (tree-based models)
- PCA or other dimensionality reduction
- Remove highly correlated features

### 2. Model Training
Recommended models to try:
- **Linear Regression** (baseline)
- **Ridge/Lasso Regression** (regularization)
- **Random Forest** (ensemble, handles non-linearity)
- **XGBoost/LightGBM** (gradient boosting)
- **Neural Networks** (deep learning)

### 3. Hyperparameter Tuning
- Grid Search or Random Search
- Cross-validation
- Use validation set for early stopping

### 4. Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

### 5. Model Interpretation
- Feature importance
- SHAP values
- Partial dependence plots

## Tips for Success

1. **Try different scalers** - Test all three scaling methods with your models
2. **Feature engineering** - The engineered features (tenure, recency, amenities) are likely very important
3. **Outliers** - If results are poor, consider more aggressive outlier removal
4. **Regularization** - Use L1/L2 regularization to prevent overfitting
5. **Ensemble methods** - Combine multiple models for better predictions
6. **Cross-validation** - Don't just rely on single train/val split

## Contact & Support

If you need to modify the cleaning pipeline:
1. Adjust imputation strategies in Step 2
2. Add more text features in Step 4
3. Try different outlier thresholds in Step 5
4. Experiment with different encoding methods in Step 7
5. Choose the best scaler for your model in Step 8

Good luck with your modeling! 🚀

