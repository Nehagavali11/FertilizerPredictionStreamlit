# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create a directory to save models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
try:
    df = pd.read_csv('fertilizer_training_dataset_3000.csv')
except FileNotFoundError:
    print("Error: 'fertilizer_training_dataset_3000.csv' not found.")
    print("Please ensure the CSV file is in the same directory as 'model_training.py'.")
    exit()

# Define features and target based on the columns identified in your CSV
numerical_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
categorical_features = ['Soil_color', 'Crop']
target = 'Fertilizer'

# Verify that all specified columns exist in the DataFrame
missing_numerical = [col for col in numerical_features if col not in df.columns]
missing_categorical = [col for col in categorical_features if col not in df.columns]
missing_target = target not in df.columns

if missing_numerical or missing_categorical or missing_target:
    print("Error: Some defined columns are missing from the dataset.")
    if missing_numerical:
        print(f"Missing numerical features: {missing_numerical}")
    if missing_categorical:
        print(f"Missing categorical features: {missing_categorical}")
    if missing_target:
        print(f"Missing target column: {target}")
    print("Available columns in CSV:", df.columns.tolist())
    exit()

# Separate features (X) and target (y)
X = df[numerical_features + categorical_features]
y = df[target]

# --- Create preprocessing pipelines with Imputation ---
# For numerical features: Impute missing values with the mean, then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Fill missing numerical values with the mean
    ('scaler', StandardScaler()) # Scale numerical features
])

# For categorical features: Impute missing values with the most frequent value, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing categorical values with the most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Create a machine learning pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the training data
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model's accuracy on the test set (optional, but good for verification)
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Save the entire trained pipeline (preprocessor + classifier)
joblib.dump(model_pipeline, 'models/fertilizer_prediction_model.joblib')

# Save the unique target classes (fertilizer names)
joblib.dump(list(y.unique()), 'models/target_classes.joblib')

# Save feature information (original numerical/categorical feature names, unique categories)
feature_info = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'soil_colors': list(df['Soil_color'].unique()),
    'crop_types': list(df['Crop'].unique())
}
joblib.dump(feature_info, 'models/feature_info.joblib')

print("Model, target classes, and feature information saved successfully in the 'models' directory.")