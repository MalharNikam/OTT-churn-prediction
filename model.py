import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Load dataset
df = pd.read_csv('data/ott_churn_dataset.csv')
df.columns = df.columns.str.strip()

# Prepare data
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})
if 'Devices' in X.columns:
    X = X.rename(columns={'Devices': 'NumOfDevices'})

# Feature groups
cat_features = ['Gender', 'SubscriptionType', 'PreferredGenre']
num_features = ['Age', 'MonthlyCharges', 'WatchTimePerDay',
                'DaysSinceLastLogin', 'NumOfComplaints', 'NumOfDevices']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

# Classifier with calibrated probabilities
rf_base = RandomForestClassifier(random_state=42)
rf = CalibratedClassifierCV(rf_base, cv=5)

# Combine pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)

# Save model and preprocessor
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("✅ Calibrated model and preprocessor saved successfully!")