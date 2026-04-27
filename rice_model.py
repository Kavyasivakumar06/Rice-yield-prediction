import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r"C:\Users\KAVIYA\Documents\rice yield\crop_production.csv")
# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())

# Filter only rice data for prediction
rice_df = df[df['Crop'].str.lower() == 'rice'].copy()
print(f"\nTotal rice records: {len(rice_df)}")

# Data Preprocessing
print("\n=== Data Preprocessing ===")

# Handle missing values
rice_df = rice_df.dropna(subset=['Production', 'Area'])
print(f"Records after removing missing values: {len(rice_df)}")

# Feature Engineering
rice_df['Yield'] = rice_df['Production'] / rice_df['Area']
rice_df = rice_df[rice_df['Yield'] > 0]  # Remove invalid yields
rice_df = rice_df[rice_df['Yield'] < rice_df['Yield'].quantile(0.99)]  # Remove outliers

print(f"Final rice records for modeling: {len(rice_df)}")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['State_Name', 'District_Name', 'Season']

for col in categorical_cols:
    le = LabelEncoder()
    rice_df[col + '_encoded'] = le.fit_transform(rice_df[col].astype(str))
    label_encoders[col] = le

# Prepare features and target
features = ['Crop_Year', 'Area', 'State_Name_encoded', 'District_Name_encoded', 'Season_encoded']
X = rice_df[features]
y = rice_df['Production']  # Predicting total production

print(f"\nFeatures used: {features}")
print(f"Target variable: Production")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Build Random Forest Model
print("\n=== Building Random Forest Model ===")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Model Evaluation
print("\n=== Model Evaluation ===")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature Importance
print("\n=== Feature Importance ===")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualization
plt.figure(figsize=(15, 10))

# 1. Actual vs Predicted
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Production')
plt.ylabel('Predicted Production')
plt.title('Actual vs Predicted Rice Production')

# 2. Feature Importance
plt.subplot(2, 3, 2)
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Random Forest')

# 3. Residuals Plot
plt.subplot(2, 3, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Production')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

# 4. Yield Distribution
plt.subplot(2, 3, 4)
plt.hist(rice_df['Yield'], bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Yield (Production/Area)')
plt.ylabel('Frequency')
plt.title('Rice Yield Distribution')

# 5. Production by Season
plt.subplot(2, 3, 5)
season_production = rice_df.groupby('Season')['Production'].mean()
season_production.plot(kind='bar', alpha=0.7)
plt.title('Average Production by Season')
plt.xticks(rotation=45)

# 6. Production Trend Over Years
plt.subplot(2, 3, 6)
yearly_production = rice_df.groupby('Crop_Year')['Production'].mean()
yearly_production.plot(kind='line', marker='o')
plt.title('Rice Production Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Average Production')

plt.tight_layout()
plt.show()

# Prediction Examples
print("\n=== Sample Predictions ===")
sample_indices = X_test.sample(5, random_state=42).index
sample_data = rice_df.loc[sample_indices]

for idx in sample_indices:
    actual = rice_df.loc[idx, 'Production']
    features_input = X.loc[idx].values.reshape(1, -1)
    predicted = rf_model.predict(features_input)[0]
    
    print(f"Year: {rice_df.loc[idx, 'Crop_Year']}, "
          f"Area: {rice_df.loc[idx, 'Area']:.2f} hectares, "
          f"Season: {rice_df.loc[idx, 'Season']}")
    print(f"  Actual Production: {actual:.2f}")
    print(f"  Predicted Production: {predicted:.2f}")
    print(f"  Error: {abs(actual - predicted):.2f}")
    print("-" * 50)

# Model Interpretation for Practical Applications
print("\n=== Practical Applications ===")
print("1. QUANTIFIED YIELD FORECASTS:")
print(f"   • Model can predict rice production with {r2*100:.1f}% accuracy")
print(f"   • Average prediction error: {rmse:.2f} units")

print("\n2. IMPROVED MANAGEMENT RECOMMENDATIONS:")
print("   • Area is the most important feature for production prediction")
print("   • Seasonal patterns significantly affect yield")

print("\n3. FOOD SECURITY PLANNING:")
print("   • Model can forecast regional production for reserve planning")
print("   • Helps identify high-risk/low-yield regions")

print("\n4. ENHANCED PRECISION AGRICULTURE:")
print("   • Optimal planting/harvesting schedules can be derived")
print("   • Identifies critical factors influencing final yield")

# Save the model for future use
import joblib
model_data = {
    'model': rf_model,
    'label_encoders': label_encoders,
    'features': features
}
joblib.dump(model_data, 'rice_yield_predictor.pkl')
print("\nModel saved as 'rice_yield_predictor.pkl'")
 
