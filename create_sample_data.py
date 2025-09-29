"""
Create sample datasets for testing the AutoML system.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Generate sample classification dataset (Iris-like)
print("Creating sample classification dataset...")
np.random.seed(42)

n_samples = 300
species = ['setosa', 'versicolor', 'virginica']

# Generate features with some realistic relationships
classification_data = []

for i, species_name in enumerate(species):
    for _ in range(n_samples // 3):
        # Create features with different distributions for each species
        if species_name == 'setosa':
            sepal_length = np.random.normal(5.0, 0.4)
            sepal_width = np.random.normal(3.4, 0.4)
            petal_length = np.random.normal(1.5, 0.2)
            petal_width = np.random.normal(0.2, 0.1)
        elif species_name == 'versicolor':
            sepal_length = np.random.normal(6.0, 0.5)
            sepal_width = np.random.normal(2.8, 0.3)
            petal_length = np.random.normal(4.2, 0.5)
            petal_width = np.random.normal(1.3, 0.2)
        else:  # virginica
            sepal_length = np.random.normal(6.5, 0.6)
            sepal_width = np.random.normal(3.0, 0.3)
            petal_length = np.random.normal(5.5, 0.6)
            petal_width = np.random.normal(2.0, 0.3)
            
        # Add some noise and missing values occasionally
        if np.random.random() < 0.05:  # 5% missing values
            sepal_length = np.nan
        if np.random.random() < 0.03:  # 3% missing values
            petal_width = np.nan
            
        classification_data.append({
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width,
            'species': species_name
        })

# Create classification DataFrame
classification_df = pd.DataFrame(classification_data)

# Add some categorical features
classification_df['habitat'] = np.random.choice(['garden', 'wild', 'greenhouse'], len(classification_df))
classification_df['season'] = np.random.choice(['spring', 'summer', 'fall'], len(classification_df))

# Save to Excel
classification_df.to_excel(data_dir / 'iris_classification.xlsx', index=False)
print(f"Created {data_dir / 'iris_classification.xlsx'}")

# Generate sample regression dataset (Housing-like)
print("Creating sample regression dataset...")
np.random.seed(42)

n_samples = 400
regression_data = []

for _ in range(n_samples):
    # Generate housing features
    bedrooms = np.random.randint(1, 6)
    bathrooms = np.random.randint(1, 4)
    sqft = np.random.normal(1800 + bedrooms * 300, 400)
    age = np.random.randint(0, 50)
    
    # Location affects price significantly
    location = np.random.choice(['downtown', 'suburb', 'rural'], p=[0.3, 0.5, 0.2])
    location_multiplier = {'downtown': 1.5, 'suburb': 1.0, 'rural': 0.7}[location]
    
    # Calculate price with some realistic relationships
    base_price = 150000 + sqft * 80 + bedrooms * 20000 + bathrooms * 15000
    age_discount = max(0, 1 - age * 0.01)  # Older houses cost less
    price = base_price * location_multiplier * age_discount
    
    # Add some noise
    price *= np.random.normal(1.0, 0.1)
    price = max(price, 50000)  # Minimum price
    
    # Add some missing values occasionally
    if np.random.random() < 0.04:  # 4% missing values
        sqft = np.nan
    if np.random.random() < 0.02:  # 2% missing values
        age = np.nan
        
    regression_data.append({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'age': age,
        'location': location,
        'garage': np.random.choice(['yes', 'no'], p=[0.7, 0.3]),
        'price': round(price, -3)  # Round to nearest thousand
    })

# Create regression DataFrame  
regression_df = pd.DataFrame(regression_data)

# Save to Excel
regression_df.to_excel(data_dir / 'housing_regression.xlsx', index=False)
print(f"Created {data_dir / 'housing_regression.xlsx'}")

print("\nDataset summary:")
print(f"Classification dataset: {classification_df.shape}")
print(f"  - Features: {list(classification_df.columns[:-1])}")
print(f"  - Target: {classification_df.columns[-1]} ({classification_df['species'].nunique()} classes)")
print(f"  - Missing values: {classification_df.isnull().sum().sum()}")

print(f"\nRegression dataset: {regression_df.shape}")  
print(f"  - Features: {list(regression_df.columns[:-1])}")
print(f"  - Target: {regression_df.columns[-1]}")
print(f"  - Missing values: {regression_df.isnull().sum().sum()}")
print(f"  - Price range: ${regression_df['price'].min():,.0f} - ${regression_df['price'].max():,.0f}")

print("\nSample datasets created successfully!")
print("You can now test the AutoML system with these datasets.")