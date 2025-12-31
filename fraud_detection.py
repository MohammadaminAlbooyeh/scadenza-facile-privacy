"""
Fraud Detection System using Rule-Based Approach
Features:
- Z-score based outlier detection using NumPy
- User-based suspicious pattern analysis
- Vectorized operations for efficiency
- GroupBy transformations for user-level statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime


class FraudDetector:
    """Rule-based fraud detection using NumPy and Pandas"""
    
    def __init__(self, z_threshold: float = 3.0):
        """
        Initialize fraud detector
        
        Args:
            z_threshold: Z-score threshold for outlier detection (default: 3.0)
        """
        self.z_threshold = z_threshold
        self.user_stats = {}
        
    def calculate_z_score(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate z-scores using vectorized NumPy operations
        
        Args:
            data: Array of numerical values
            
        Returns:
            Array of z-scores
        """
        mean = np.mean(data)
        std = np.std(data)
        
        # Avoid division by zero
        if std == 0:
            return np.zeros_like(data)
        
        # Vectorized z-score calculation
        z_scores = (data - mean) / std
        return z_scores
    
    def detect_amount_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect transactions with anomalous amounts using z-score
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with z-scores and outlier flags
        """
        # Vectorized z-score calculation
        df['amount_zscore'] = self.calculate_z_score(df['amount'].values)
        
        # Flag outliers
        df['amount_outlier'] = np.abs(df['amount_zscore']) > self.z_threshold
        
        return df
    
    def detect_user_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect suspicious patterns per user using GroupBy + transform
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with user-level statistics and anomaly flags
        """
        # Calculate user-level statistics using groupby + transform
        df['user_mean_amount'] = df.groupby('user_id')['amount'].transform('mean')
        df['user_std_amount'] = df.groupby('user_id')['amount'].transform('std')
        df['user_transaction_count'] = df.groupby('user_id')['transaction_id'].transform('count')
        
        # Calculate deviation from user's typical behavior
        df['user_amount_deviation'] = np.abs(df['amount'] - df['user_mean_amount'])
        
        # Replace NaN std with 0 for users with single transaction
        df['user_std_amount'].fillna(0, inplace=True)
        
        # Calculate user-specific z-score
        df['user_zscore'] = np.where(
            df['user_std_amount'] > 0,
            (df['amount'] - df['user_mean_amount']) / df['user_std_amount'],
            0
        )
        
        # Flag suspicious transactions based on user behavior
        df['user_pattern_anomaly'] = np.abs(df['user_zscore']) > self.z_threshold
        
        return df
    
    def detect_location_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect unusual locations for each user
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with location-based anomaly flags
        """
        # Get most common location per user
        df['user_common_location'] = df.groupby('user_id')['location'].transform(
            lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
        )
        
        # Flag transactions from unusual locations
        df['location_anomaly'] = df['location'] != df['user_common_location']
        
        # Count unique locations per user
        df['user_unique_locations'] = df.groupby('user_id')['location'].transform('nunique')
        
        return df
    
    def detect_time_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect unusual transaction times using vectorized operations
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with time-based anomaly flags
        """
        # Extract hour from timestamp
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Calculate time-based statistics per user
        df['user_mean_hour'] = df.groupby('user_id')['hour'].transform('mean')
        df['user_std_hour'] = df.groupby('user_id')['hour'].transform('std')
        
        # Replace NaN std with 0
        df['user_std_hour'].fillna(0, inplace=True)
        
        # Calculate hour deviation (circular distance)
        hour_diff = np.abs(df['hour'] - df['user_mean_hour'])
        df['hour_deviation'] = np.minimum(hour_diff, 24 - hour_diff)
        
        # Flag unusual transaction times (more than 6 hours from mean)
        df['time_anomaly'] = df['hour_deviation'] > 6
        
        return df
    
    def calculate_fraud_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite fraud score based on multiple factors
        
        Args:
            df: DataFrame with all anomaly flags
            
        Returns:
            DataFrame with fraud scores and predictions
        """
        # Weighted fraud score (vectorized)
        weights = {
            'amount_outlier': 2.0,
            'user_pattern_anomaly': 3.0,
            'location_anomaly': 1.5,
            'time_anomaly': 1.0
        }
        
        # Calculate weighted sum using vectorized operations
        fraud_score = (
            df['amount_outlier'].astype(float) * weights['amount_outlier'] +
            df['user_pattern_anomaly'].astype(float) * weights['user_pattern_anomaly'] +
            df['location_anomaly'].astype(float) * weights['location_anomaly'] +
            df['time_anomaly'].astype(float) * weights['time_anomaly']
        )
        
        df['fraud_score'] = fraud_score
        
        # Normalize to 0-100 scale
        max_score = sum(weights.values())
        df['fraud_score_normalized'] = (df['fraud_score'] / max_score) * 100
        
        # Flag as fraudulent if score exceeds threshold
        df['is_fraud'] = df['fraud_score'] >= 4.0  # At least 2 high-weight flags
        
        return df
    
    def detect_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete fraud detection pipeline
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with all fraud detection features and predictions
        """
        # Make a copy to avoid modifying original
        df_result = df.copy()
        
        # Run all detection methods
        df_result = self.detect_amount_outliers(df_result)
        df_result = self.detect_user_patterns(df_result)
        df_result = self.detect_location_anomalies(df_result)
        df_result = self.detect_time_anomalies(df_result)
        df_result = self.calculate_fraud_score(df_result)
        
        return df_result
    
    def get_fraud_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics of fraud detection results
        
        Args:
            df: DataFrame with fraud detection results
            
        Returns:
            Dictionary with summary statistics
        """
        total_transactions = len(df)
        fraudulent_transactions = df['is_fraud'].sum()
        fraud_rate = (fraudulent_transactions / total_transactions) * 100
        
        summary = {
            'total_transactions': total_transactions,
            'fraudulent_transactions': int(fraudulent_transactions),
            'fraud_rate': f'{fraud_rate:.2f}%',
            'amount_outliers': int(df['amount_outlier'].sum()),
            'user_pattern_anomalies': int(df['user_pattern_anomaly'].sum()),
            'location_anomalies': int(df['location_anomaly'].sum()),
            'time_anomalies': int(df['time_anomaly'].sum()),
            'avg_fraud_score': f'{df["fraud_score_normalized"].mean():.2f}',
            'max_fraud_score': f'{df["fraud_score_normalized"].max():.2f}'
        }
        
        return summary
    
    def get_high_risk_users(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Identify users with highest fraud rates using efficient filtering
        
        Args:
            df: DataFrame with fraud detection results
            top_n: Number of top users to return
            
        Returns:
            DataFrame with high-risk users
        """
        # Group by user and calculate fraud statistics (vectorized aggregation)
        user_risk = df.groupby('user_id').agg({
            'transaction_id': 'count',
            'is_fraud': ['sum', 'mean'],
            'fraud_score_normalized': 'mean'
        }).reset_index()
        
        # Flatten column names
        user_risk.columns = ['user_id', 'total_transactions', 'fraud_count', 
                            'fraud_rate', 'avg_fraud_score']
        
        # Efficient filtering: users with at least 1 fraud
        user_risk = user_risk[user_risk['fraud_count'] > 0]
        
        # Sort by fraud rate and get top N
        user_risk = user_risk.sort_values('fraud_rate', ascending=False).head(top_n)
        
        return user_risk


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("FRAUD DETECTION SYSTEM - Rule-Based Approach")
    print("=" * 70)
    print()
    
    # Load or generate data
    print("Loading transaction data...")
    try:
        # Try to load existing data
        df = pd.read_csv('transactions.csv')
        print(f"✓ Loaded {len(df)} transactions from transactions.csv")
    except FileNotFoundError:
        print("⚠ transactions.csv not found. Please run generate_data.py first.")
        print("  Or create sample data manually.")
        return
    
    print()
    
    # Initialize fraud detector
    detector = FraudDetector(z_threshold=3.0)
    
    # Run fraud detection
    print("Running fraud detection pipeline...")
    print("  - Detecting amount outliers (z-score)...")
    print("  - Analyzing user patterns (GroupBy + transform)...")
    print("  - Checking location anomalies...")
    print("  - Detecting time anomalies...")
    print("  - Calculating fraud scores...")
    
    results = detector.detect_fraud(df)
    print("✓ Fraud detection complete!")
    print()
    
    # Display summary
    print("-" * 70)
    print("DETECTION SUMMARY")
    print("-" * 70)
    summary = detector.get_fraud_summary(results)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title():.<40} {value:>20}")
    print()
    
    # Display high-risk users
    print("-" * 70)
    print("TOP 10 HIGH-RISK USERS")
    print("-" * 70)
    high_risk = detector.get_high_risk_users(results, top_n=10)
    print(high_risk.to_string(index=False))
    print()
    
    # Display sample fraudulent transactions
    print("-" * 70)
    print("SAMPLE FRAUDULENT TRANSACTIONS")
    print("-" * 70)
    fraud_transactions = results[results['is_fraud']].head(10)
    display_cols = ['transaction_id', 'user_id', 'amount', 'location', 
                   'fraud_score_normalized', 'is_fraud']
    print(fraud_transactions[display_cols].to_string(index=False))
    print()
    
    # Save results
    output_file = 'fraud_detection_results.csv'
    results.to_csv(output_file, index=False)
    print(f"✓ Results saved to {output_file}")
    print()
    
    # Display statistics using NumPy
    print("-" * 70)
    print("STATISTICAL ANALYSIS (NumPy)")
    print("-" * 70)
    fraud_amounts = results[results['is_fraud']]['amount'].values
    normal_amounts = results[~results['is_fraud']]['amount'].values
    
    print(f"Fraudulent Transactions:")
    print(f"  Mean Amount.............. ${np.mean(fraud_amounts):,.2f}")
    print(f"  Median Amount............ ${np.median(fraud_amounts):,.2f}")
    print(f"  Std Deviation............ ${np.std(fraud_amounts):,.2f}")
    print(f"  Min/Max.................. ${np.min(fraud_amounts):,.2f} / ${np.max(fraud_amounts):,.2f}")
    print()
    print(f"Normal Transactions:")
    print(f"  Mean Amount.............. ${np.mean(normal_amounts):,.2f}")
    print(f"  Median Amount............ ${np.median(normal_amounts):,.2f}")
    print(f"  Std Deviation............ ${np.std(normal_amounts):,.2f}")
    print(f"  Min/Max.................. ${np.min(normal_amounts):,.2f} / ${np.max(normal_amounts):,.2f}")
    print()
    
    print("=" * 70)
    print("Detection Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
