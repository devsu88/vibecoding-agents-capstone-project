import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def load_titanic_data():
    # Helper function to generate synthetic data matching the schema
    n_samples = 100
    data = {
        'PassengerId': np.arange(n_samples),
        'Survived': np.random.randint(0, 2, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.4, 0.4]),
        'Name': [f'Mr. Test {i}' for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.uniform(18, 65, n_samples),
        'SibSp': np.random.randint(0, 4, n_samples),
        'Parch': np.random.randint(0, 3, n_samples),
        'Ticket': [f'TKT{i}' for i in range(n_samples)],
        'Fare': np.random.lognormal(mean=3.5, sigma=0.8, size=n_samples),
        'Cabin': np.random.choice([f'C{i}' for i in range(10)] + [np.nan], n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q', np.nan], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
    }
    df = pd.DataFrame(data)

    # Introduce missing values for demonstration
    df.loc[df.sample(frac=0.1, random_state=42).index, 'Age'] = np.nan
    df.loc[df.sample(frac=0.02, random_state=42).index, 'Fare'] = np.nan

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y

# Define feature sets
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Pclass', 'Embarked']

# Create numerical pipeline (Imputation -> Scaling)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create categorical pipeline (Imputation -> Encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    # Drop columns not explicitly handled (like Name, Ticket, Cabin, PassengerId)
    remainder='drop'
)

# 1. Random Forest Pipeline (Non-linear Ensemble Baseline)
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# 2. MLP Classifier Pipeline (Neural Network Baseline)
# Addressing the human feedback requesting a neural network.
pipeline_mlp = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True))
])


if __name__ == '__main__':
    # Load synthetic data for verification
    X, y = load_titanic_data()

    # Define the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("--- Starting Baseline Model Evaluation (5-Fold CV Accuracy) ---")

    # Evaluate Random Forest Classifier
    print("Evaluating Random Forest Classifier...")
    scores_rf = cross_val_score(
        pipeline_rf, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1
    )
    results["RandomForestClassifier_Mean_Accuracy"] = np.mean(scores_rf)
    results["RandomForestClassifier_Std_Accuracy"] = np.std(scores_rf)

    # Evaluate MLP Classifier
    print("Evaluating MLP Classifier (Neural Network)...")
    scores_mlp = cross_val_score(
        pipeline_mlp, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1
    )
    results["MLPClassifier_Mean_Accuracy"] = np.mean(scores_mlp)
    results["MLPClassifier_Std_Accuracy"] = np.std(scores_mlp)

    print("\nCross-Validation Results:")
    print("==================================================")
    for model in ["RandomForestClassifier", "MLPClassifier"]:
        mean_score = results[f"{model}_Mean_Accuracy"]
        std_score = results[f"{model}_Std_Accuracy"]
        print(f"{model}:")
        print(f"  Mean Accuracy: {mean_score:.4f}")
        print(f"  Std Deviation: {std_score:.4f}")
    print("==================================================")