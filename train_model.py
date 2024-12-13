import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('credit_risk_dataset.csv')

# Check for missing values in the dataset
print("Missing values in each column:")
print(df.isnull().sum())

# Fill missing numerical data using SimpleImputer (mean strategy)
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing numerical data
    ('scaler', StandardScaler())  # Scale numerical data
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical data
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
])

# Combine both transformers into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare the feature matrix and target variable
X = df.drop('loan_status', axis=1)  # Features
y = df['loan_status']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that first applies preprocessing, then trains the RandomForest model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply preprocessing
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest Classifier
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict on test data to evaluate the model
y_pred = model_pipeline.predict(X_test)

# Print classification report and accuracy score
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the trained model to a pickle file
with open('loan_model.pkl', 'wb') as model_file:
    pickle.dump(model_pipeline, model_file)

print("Model trained and saved as 'loan_model.pkl'")
