"""
Generate synthetic transaction data for fraud detection
Creates realistic patterns with both normal and fraudulent transactions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_transactions(n_users: int = 100, 
                         n_transactions: int = 10000,
                         fraud_rate: float = 0.05,
                         random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transaction data with normal and fraudulent patterns
    
    Args:
        n_users: Number of unique users
        n_transactions: Total number of transactions to generate
        fraud_rate: Proportion of fraudulent transactions (0-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with transaction data
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print(f"Generating {n_transactions} transactions for {n_users} users...")
    print(f"Target fraud rate: {fraud_rate * 100:.1f}%")
    print()
    
    # Generate base transaction data
    transactions = []
    
    # Define locations
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                'London', 'Paris', 'Tokyo', 'Sydney', 'Toronto']
    
    # Create user profiles (each user has typical behavior)
    user_profiles = {}
    for user_id in range(1, n_users + 1):
        user_profiles[user_id] = {
            'typical_amount_mean': np.random.uniform(50, 500),
            'typical_amount_std': np.random.uniform(20, 100),
            'primary_location': random.choice(locations[:10]),  # Domestic
            'typical_hour_mean': np.random.randint(8, 20),  # 8am - 8pm
            'typical_hour_std': 3
        }
    
    # Start date for transactions
    start_date = datetime.now() - timedelta(days=90)
    
    # Calculate number of fraudulent transactions
    n_fraud = int(n_transactions * fraud_rate)
    n_normal = n_transactions - n_fraud
    
    print("Generating normal transactions...")
    # Generate normal transactions
    for i in range(n_normal):
        user_id = random.randint(1, n_users)
        profile = user_profiles[user_id]
        
        # Generate amount based on user's typical behavior
        amount = np.random.normal(profile['typical_amount_mean'], 
                                 profile['typical_amount_std'])
        amount = max(10, amount)  # Minimum $10
        
        # Generate timestamp
        days_offset = np.random.randint(0, 90)
        hour = int(np.random.normal(profile['typical_hour_mean'], 
                                    profile['typical_hour_std']))
        hour = max(0, min(23, hour))  # Keep within 0-23
        minute = np.random.randint(0, 60)
        
        timestamp = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
        
        # Use primary location (90% of time) or nearby location (10%)
        if np.random.random() < 0.9:
            location = profile['primary_location']
        else:
            # Occasional transaction from nearby location
            location = random.choice(locations[:10])
        
        transactions.append({
            'transaction_id': f'TXN{i+1:06d}',
            'user_id': f'USER{user_id:04d}',
            'amount': round(amount, 2),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'location': location,
            'is_fraud_actual': False
        })
    
    print("Generating fraudulent transactions...")
    # Generate fraudulent transactions with anomalous patterns
    for i in range(n_fraud):
        user_id = random.randint(1, n_users)
        profile = user_profiles[user_id]
        
        # Choose fraud type
        fraud_type = random.choice(['amount', 'location', 'time', 'combined'])
        
        # Generate anomalous transaction based on type
        if fraud_type == 'amount':
            # Unusually large amount (5-10x typical)
            amount = profile['typical_amount_mean'] * np.random.uniform(5, 10)
            
            # Otherwise normal patterns
            days_offset = np.random.randint(0, 90)
            hour = int(np.random.normal(profile['typical_hour_mean'], 
                                       profile['typical_hour_std']))
            hour = max(0, min(23, hour))
            location = profile['primary_location']
            
        elif fraud_type == 'location':
            # Normal amount but unusual location (international)
            amount = np.random.normal(profile['typical_amount_mean'], 
                                     profile['typical_amount_std'])
            amount = max(10, amount)
            
            # International location
            location = random.choice(locations[10:])
            
            # Normal time
            days_offset = np.random.randint(0, 90)
            hour = int(np.random.normal(profile['typical_hour_mean'], 
                                       profile['typical_hour_std']))
            hour = max(0, min(23, hour))
            
        elif fraud_type == 'time':
            # Normal amount but unusual time (late night/early morning)
            amount = np.random.normal(profile['typical_amount_mean'], 
                                     profile['typical_amount_std'])
            amount = max(10, amount)
            
            # Unusual hour (midnight to 5am)
            hour = np.random.randint(0, 6)
            
            # Normal location
            days_offset = np.random.randint(0, 90)
            location = profile['primary_location']
            
        else:  # combined - multiple anomalies
            # Large amount
            amount = profile['typical_amount_mean'] * np.random.uniform(5, 10)
            
            # Foreign location
            location = random.choice(locations[10:])
            
            # Unusual hour
            hour = np.random.randint(0, 6)
            days_offset = np.random.randint(0, 90)
        
        minute = np.random.randint(0, 60)
        timestamp = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
        
        transactions.append({
            'transaction_id': f'TXN{n_normal + i + 1:06d}',
            'user_id': f'USER{user_id:04d}',
            'amount': round(amount, 2),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'location': location,
            'is_fraud_actual': True
        })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Shuffle transactions
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✓ Generated {len(df)} transactions")
    print(f"  - Normal: {(~df['is_fraud_actual']).sum()}")
    print(f"  - Fraudulent: {df['is_fraud_actual'].sum()}")
    print()
    
    return df


def display_data_statistics(df: pd.DataFrame):
    """Display statistics about the generated data"""
    print("=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    print()
    
    print(f"Total Transactions: {len(df)}")
    print(f"Unique Users: {df['user_id'].nunique()}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Locations: {df['location'].nunique()} unique locations")
    print()
    
    print("Amount Statistics:")
    print(f"  Mean: ${df['amount'].mean():,.2f}")
    print(f"  Median: ${df['amount'].median():,.2f}")
    print(f"  Std Dev: ${df['amount'].std():,.2f}")
    print(f"  Range: ${df['amount'].min():,.2f} - ${df['amount'].max():,.2f}")
    print()
    
    print("Fraud Statistics:")
    fraud_count = df['is_fraud_actual'].sum()
    fraud_rate = (fraud_count / len(df)) * 100
    print(f"  Fraudulent Transactions: {fraud_count}")
    print(f"  Fraud Rate: {fraud_rate:.2f}%")
    print()
    
    print("Top 5 Locations:")
    top_locations = df['location'].value_counts().head()
    for loc, count in top_locations.items():
        print(f"  {loc:.<30} {count:>5} ({count/len(df)*100:.1f}%)")
    print()
    
    print("Top 10 Users by Transaction Count:")
    top_users = df['user_id'].value_counts().head(10)
    for user, count in top_users.items():
        fraud_count_user = df[df['user_id'] == user]['is_fraud_actual'].sum()
        print(f"  {user}: {count} transactions ({fraud_count_user} fraudulent)")
    print()


def main():
    """Main execution function"""
    print("=" * 70)
    print("TRANSACTION DATA GENERATOR")
    print("=" * 70)
    print()
    
    # Generate data
    df = generate_transactions(
        n_users=100,
        n_transactions=10000,
        fraud_rate=0.05,  # 5% fraud rate
        random_seed=42
    )
    
    # Display statistics
    display_data_statistics(df)
    
    # Save to CSV
    output_file = 'transactions.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Data saved to {output_file}")
    print()
    
    # Display sample transactions
    print("=" * 70)
    print("SAMPLE TRANSACTIONS")
    print("=" * 70)
    print()
    print("Normal Transactions:")
    print(df[~df['is_fraud_actual']].head(5).to_string(index=False))
    print()
    print("Fraudulent Transactions:")
    print(df[df['is_fraud_actual']].head(5).to_string(index=False))
    print()
    
    print("=" * 70)
    print("Data generation complete!")
    print(f"Run 'python fraud_detection.py' to detect fraud in {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
