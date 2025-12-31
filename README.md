# Fraud Detection System (Rule-Based)

A comprehensive fraud detection system using rule-based approaches with NumPy and Pandas. This project demonstrates vectorized anomaly detection, efficient filtering, and user-based pattern analysis for identifying fraudulent transactions.

## 🎯 Features

### Detection Methods
- **Z-Score Outlier Detection**: Identifies transactions with anomalous amounts using statistical methods
- **User Pattern Analysis**: Detects deviations from individual user behavior patterns
- **Location Anomalies**: Flags transactions from unusual locations
- **Time-Based Anomalies**: Identifies transactions at suspicious times
- **Composite Fraud Scoring**: Weighted scoring system combining multiple signals

### Technical Skills Demonstrated
- ✅ **Vectorized Operations**: Efficient NumPy array operations for large-scale data processing
- ✅ **GroupBy + Transform**: Pandas groupby with transform for user-level statistics
- ✅ **Efficient Filtering**: Optimized data filtering and aggregation techniques
- ✅ **Statistical Analysis**: Z-score calculations, standard deviations, and distributions

## 📊 Data Structure

Each transaction contains:
- `transaction_id`: Unique identifier
- `user_id`: User identifier
- `amount`: Transaction amount
- `timestamp`: Transaction date and time
- `location`: Transaction location
- `is_fraud_actual`: Ground truth label (for evaluation)

## 🚀 Quick Start

### Installation

1. Clone or download this repository:
```bash
cd fraud_detection_numpy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Generate Sample Data
```bash
python generate_data.py
```

This creates `transactions.csv` with 10,000 synthetic transactions (5% fraud rate).

**Output:**
- Transaction dataset with realistic patterns
- Mix of normal and fraudulent transactions
- Data statistics and sample preview

#### Step 2: Run Fraud Detection
```bash
python fraud_detection.py
```

This analyzes transactions and generates a fraud detection report.

**Output:**
- Detection summary statistics
- High-risk users identification
- Sample fraudulent transactions
- Statistical analysis (NumPy)
- Saved results in `fraud_detection_results.csv`

## 📈 Detection Methodology

### 1. Amount Outlier Detection (Z-Score)
```python
z_score = (amount - mean) / std_dev
is_outlier = |z_score| > 3.0
```

Uses vectorized NumPy operations to calculate z-scores for all transactions, flagging those with z-scores exceeding the threshold.

### 2. User Pattern Analysis (GroupBy + Transform)
```python
# Calculate user-specific statistics
user_mean = df.groupby('user_id')['amount'].transform('mean')
user_std = df.groupby('user_id')['amount'].transform('std')
user_zscore = (amount - user_mean) / user_std
```

Identifies transactions that deviate significantly from each user's typical behavior using groupby transformations.

### 3. Location Anomaly Detection
- Identifies each user's most common location
- Flags transactions from unusual/foreign locations
- Tracks number of unique locations per user

### 4. Time Anomaly Detection
- Extracts hour from timestamp
- Calculates user's typical transaction hours
- Flags transactions outside normal time windows (>6 hours deviation)

### 5. Composite Fraud Score
Weighted combination of all signals:
- Amount outliers: 2.0 weight
- User pattern anomalies: 3.0 weight
- Location anomalies: 1.5 weight
- Time anomalies: 1.0 weight

Transactions with score ≥ 4.0 flagged as fraudulent.

## 📁 Project Structure

```
fraud_detection_numpy/
├── fraud_detection.py      # Main fraud detection system
├── generate_data.py         # Synthetic data generator
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── transactions.csv        # Generated transaction data (after running generate_data.py)
└── fraud_detection_results.csv  # Detection results (after running fraud_detection.py)
```

## 🔍 Example Output

```
======================================================================
FRAUD DETECTION SYSTEM - Rule-Based Approach
======================================================================

Loading transaction data...
✓ Loaded 10000 transactions from transactions.csv

Running fraud detection pipeline...
  - Detecting amount outliers (z-score)...
  - Analyzing user patterns (GroupBy + transform)...
  - Checking location anomalies...
  - Detecting time anomalies...
  - Calculating fraud scores...
✓ Fraud detection complete!

----------------------------------------------------------------------
DETECTION SUMMARY
----------------------------------------------------------------------
Total Transactions..........................           10000
Fraudulent Transactions.....................             512
Fraud Rate..................................           5.12%
Amount Outliers.............................             324
User Pattern Anomalies......................             447
Location Anomalies..........................             289
Time Anomalies..............................             198
Avg Fraud Score.............................           18.45
Max Fraud Score.............................          100.00
```

## 🛠️ Customization

### Adjust Z-Score Threshold
```python
detector = FraudDetector(z_threshold=2.5)  # More sensitive
detector = FraudDetector(z_threshold=4.0)  # Less sensitive
```

### Modify Fraud Score Weights
Edit the `calculate_fraud_score()` method in `fraud_detection.py`:
```python
weights = {
    'amount_outlier': 2.0,
    'user_pattern_anomaly': 3.0,
    'location_anomaly': 1.5,
    'time_anomaly': 1.0
}
```

### Generate Different Data Patterns
Modify parameters in `generate_data.py`:
```python
df = generate_transactions(
    n_users=200,           # More users
    n_transactions=50000,  # More transactions
    fraud_rate=0.10,       # Higher fraud rate
    random_seed=123        # Different random seed
)
```

## 📊 Performance Characteristics

- **Vectorized Operations**: All calculations use NumPy/Pandas vectorization for speed
- **Memory Efficient**: Uses in-place operations where possible
- **Scalable**: Can process 100K+ transactions efficiently
- **Time Complexity**: O(n) for most operations, O(n log n) for sorting

## 🎓 Learning Outcomes

This project demonstrates:
1. **Statistical Analysis**: Z-score calculations for outlier detection
2. **Pandas GroupBy**: Advanced aggregation with transform
3. **NumPy Vectorization**: Efficient array operations
4. **Data Processing**: Cleaning, filtering, and transformation
5. **Feature Engineering**: Creating fraud indicators from raw data
6. **Rule-Based Systems**: Combining multiple signals for decision making

## 📝 Future Enhancements

Potential improvements:
- [ ] Add velocity checks (multiple transactions in short time)
- [ ] Implement machine learning models (Random Forest, XGBoost)
- [ ] Real-time streaming detection
- [ ] Visualization dashboard
- [ ] Cross-user pattern detection (account takeover)
- [ ] Merchant category analysis
- [ ] Device fingerprinting

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and improve this project. Suggestions and contributions are welcome!

---

**Author**: Fraud Detection System  
**Date**: 2025  
**Purpose**: Educational demonstration of rule-based fraud detection
