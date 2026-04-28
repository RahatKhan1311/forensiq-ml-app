import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# STEP 1: CREATE DATA
np.random.seed(42)
n = 1000

locations = ['Mumbai','Delhi','Pune','Bangalore','Chennai']
evidence_types = ['CCTV', 'Logs', 'Witness', 'Documents', 'Forensic']
crime_types = ['Theft', 'Cyber Crime', 'Assault', 'Fraud', 'Robbery']
statuses = ['Open', 'Closed', 'Pending']

data = []

for _ in range(n):
    location = np.random.choice(locations)
    evidence_type = np.random.choice(evidence_types)
    status = np.random.choice(statuses)
    month = np.random.randint(1, 13)
    weekday = np.random.randint(0, 7)
    
    # realistic pattern
    if evidence_type == 'Logs':
        crime_type = np.random.choice(['Cyber Crime', 'Fraud'], p=[0.7, 0.3])
    elif evidence_type == 'CCTV':
        crime_type = np.random.choice(['Theft', 'Robbery', 'Assault'], p=[0.5, 0.3, 0.2])
    elif evidence_type == 'Witness':
        crime_type = np.random.choice(['Assault', 'Robbery', 'Theft'], p=[0.6, 0.2, 0.2])
    elif evidence_type == 'Documents':
        crime_type = np.random.choice(['Fraud', 'Cyber Crime'], p=[0.8, 0.2])
    else:
        crime_type = np.random.choice(['Assault', 'Robbery'], p=[0.6, 0.4])
    
    data.append([location, evidence_type, status, month, weekday, crime_type])

df = pd.DataFrame(data, columns=['location','evidence_type','status','month','weekday','crime_type'])

# STEP 2: SPLIT FEATURES
X = df[['location','evidence_type','status','month','weekday']]
y = df['crime_type']

# STEP 3: PIPELINE 
categorical_cols = ['location','evidence_type','status']
numerical_cols = ['month','weekday']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# STEP 4: TRAIN
model.fit(X, y)

# STEP 5: SAVE MODEL
joblib.dump(model, "crime_model.pkl")

print("✅ Model trained and saved as crime_model.pkl")