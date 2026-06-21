import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.datasets import make_classification

# --- 1. Data Generation ---
# Create synthetic data matching the expected schema for multi-class classification
# with mixed numerical/categorical features and introducing NaNs for pipeline testing.

def generate_synthetic_data(n_samples=1000):
    # Generate base synthetic numerical features and a multi-class target
    X_synthetic, y_synthetic = make_classification(
        n_samples=n_samples,
        n_features=5, # Features for classification
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # Define feature names based on the astronomical context
    numerical_features_synth = ['Temperature', 'Luminosity', 'Magnitude']
    categorical_features_synth = ['Color', 'Band']

    X = pd.DataFrame(X_synthetic[:, :3], columns=numerical_features_synth)
    
    # Generate synthetic categorical features
    np.random.seed(42)
    X['Color'] = np.random.choice(['Red', 'Blue', 'Yellow', 'Unknown'], size=n_samples)
    X['Band'] = np.random.choice(['UV', 'Visible', 'IR'], size=n_samples)
    
    # Introduce some missing values to test the imputers
    X.loc[50:60, 'Temperature'] = np.nan
    X.loc[100:110, 'Color'] = np.nan

    # Target variable mapping
    target_map = {0: 'Star', 1: 'Galaxy', 2: 'QSO'}
    y = pd.Series(y_synthetic, name='Stellar_Class').map(target_map)
    
    return X, y, numerical_features_synth, categorical_features_synth

if __name__ == '__main__':
    X, y, numerical_features, categorical_features = generate_synthetic_data()
    print(f"Synthetic data generated: X shape {X.shape}, y shape {y.shape}")

    # --- 2. Preprocessing Definition ---

    # Pipeline for numerical features (Impute median, then Scale)
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features (Impute most frequent, then One-Hot Encode)
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough')

    # --- 3. Baseline Models ---

    # 1. High-Performance Baseline: HistGradientBoostingClassifier
    hgbc_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])

    # 2. Robust Baseline: RandomForestClassifier
    rfc_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1))
    ])

    baselines = {
        'HistGradientBoostingClassifier': hgbc_pipeline,
        'RandomForestClassifier': rfc_pipeline
    }

    # --- 4. Evaluation ---

    N_SPLITS = 5
    cv_strategy = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'neg_log_loss'] 
    evaluation_results = {}

    print(f"\nStarting {N_SPLITS}-Fold Stratified Cross-Validation on synthetic data (N={len(X)})...")

    # Iterating through baseline models
    for model_name, pipeline in baselines.items():
        print(f"\nEvaluating: {model_name}")
        
        # Use cross_validate to get comprehensive scores and timing
        scores = cross_validate(
            estimator=pipeline, 
            X=X, 
            y=y, 
            cv=cv_strategy, 
            scoring=scoring_metrics, 
            n_jobs=-1,
            return_train_score=True
        )
        
        # Store and print results
        evaluation_results[model_name] = {
            'Test_LogLoss_Mean': np.mean(scores['test_neg_log_loss']),
            'Test_LogLoss_Std': np.std(scores['test_neg_log_loss']),
            'Test_Accuracy_Mean': np.mean(scores['test_accuracy']),
            'Fit_Time_Mean': np.mean(scores['fit_time']),
        }
        
        print(f"--- Results for {model_name} ---")
        print(f"Mean Test Log Loss: {-evaluation_results[model_name]['Test_LogLoss_Mean']:.4f} (+/- {evaluation_results[model_name]['Test_LogLoss_Std']:.4f})")
        print(f"Mean Test Accuracy: {evaluation_results[model_name]['Test_Accuracy_Mean']:.4f}")