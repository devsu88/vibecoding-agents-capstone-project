## Final Project Summary: MNIST Digit Recognition

This project addresses the multi-class image classification task of recognizing handwritten digits (0-9) using the MNIST dataset, formalized as a Kaggle competition challenge. The raw input consists of 784 pixel intensity features for 28x28 grayscale images.

### Methodology and Pipeline Construction

1.  **Preprocessing**: All 784 pixel features (initially 0-255) were scaled to the range [0.0, 1.0] using `MinMaxScaler`. This normalization is critical for optimizing gradient-based models like XGBoost and neural networks.
2.  **Feature Engineering**: Two simple aggregate features were derived from the normalized pixel data to provide non-spatial context, primarily benefiting the non-CNN baseline:
    *   **Total Intensity**: Sum of all pixel values.
    *   **Pixel Density**: Count of pixels above a minimal threshold (0.01).
3.  **Modeling Strategy (Baselines)**:
    *   **XGBoost Classifier**: A robust traditional machine learning model was chosen, integrated into a `Pipeline` encompassing scaling, feature engineering, and classification. This model serves as an efficient and interpretable baseline.
    *   **Convolutional Neural Network (CNN)**: A simple LeNet-inspired architecture was defined, which is the state-of-the-art approach for image data, requiring raw pixel normalization and one-hot encoding of labels.
4.  **Evaluation**: The XGBoost Pipeline performance was rigorously assessed using Stratified 5-Fold Cross-Validation, ensuring balanced representation of all 10 digit classes across folds. The primary metric is classification accuracy.

### Execution and Results

The full runnable script loads `train.csv` and `test.csv`. The XGBoost pipeline successfully integrates the custom feature engineering step with the classification model. The cross-validation results provide a reliable estimate of the XGBoost baseline performance on the training data.

*Note: The CNN model requires substantial training time (many epochs) to achieve competitive performance. The script includes the model definition and a quick 1-epoch training demonstration for structural completeness.*

### Complete Python Script

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape


class ImageFeatureGenerator(BaseEstimator, TransformerMixin):
    """Calculates simple aggregate features (Total Intensity and Density) from normalized pixel data."""
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # X is expected to be normalized data (N, 784) in the range [0, 1]
        
        # 1. Total Intensity (Sum of all pixels)
        total_intensity = np.sum(X, axis=1, keepdims=True)
        
        # 2. Pixel Density (Count of pixels above threshold)
        pixel_density = np.sum(X > self.threshold, axis=1, keepdims=True)
        
        # Concatenate the new features to the original pixel data
        X_transformed = np.hstack([X, total_intensity, pixel_density])
        
        return X_transformed


def create_cnn_model(input_dim=(784,), output_shape=(28, 28, 1), num_classes=10):
    """Defines the Simple CNN (LeNet-inspired) model."""
    model = Sequential([
        # Reshape the flattened 784 features into a 2D image (28x28x1)
        Reshape(output_shape, input_shape=input_dim),
        
        # Conv Block 1
        Conv2D(32, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Conv Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Classification Head
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Define the XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    n_estimators=150,
    learning_rate=0.1,
    max_depth=7, 
    use_label_encoder=False,
    eval_metric='merror', 
    n_jobs=-1,
    random_state=42
)

# Define the full pipeline: Scale -> Feature Gen -> Model
xgb_pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler()), 
    ('feature_gen', ImageFeatureGenerator(threshold=0.01)), 
    ('classifier', xgb_model)
])


if __name__ == '__main__':
    # Load Data
    try:
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv') 
        print(f"Training data shape: {df_train.shape}")
        
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found. Please ensure data files are present in the directory.")
        exit()

    
    # Separate features (X) and target (y)
    X = df_train.drop('label', axis=1).values
    y = df_train['label'].values
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # --- Evaluation: XGBoost Pipeline using Stratified K-Fold ---
    
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    print(f"\nStarting Stratified {N_SPLITS}-Fold Cross-Validation for XGBoost Pipeline...")

    # Use cross_val_score on the defined pipeline, which includes scaling and feature engineering
    xgb_scores = cross_val_score(
        xgb_pipeline,
        X, y,
        cv=skf,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    print(f"\nXGBoost Pipeline Model Accuracy Scores (Stratified {N_SPLITS} Folds):")
    print(xgb_scores)
    print(f"Mean Accuracy: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")
    
    # --- Example Training for CNN ---
    
    print("\nPreparing data for CNN Model Training...")

    # Normalize X (0-255 -> 0-1)
    X_cnn = X / 255.0
    
    # One-hot encode y
    y_cnn = tf.keras.utils.to_categorical(y, num_classes=10)
    
    # Split for simple CNN training demonstration
    X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
        X_cnn, y_cnn, test_size=0.1, random_state=42, stratify=y
    )
    
    # Instantiate and train the CNN model 
    cnn_model = create_cnn_model()
    
    print("Training CNN Model (1 Epoch for quick demo)... A full training requires many epochs.")
    history = cnn_model.fit(
        X_train_cnn, y_train_cnn,
        epochs=1, 
        batch_size=128,
        validation_data=(X_val_cnn, y_val_cnn),
        verbose=0
    )
    
    print(f"CNN Validation Accuracy (1 Epoch): {history.history['val_accuracy'][0]:.4f}")
    
    # --- Prediction Example ---
    
    X_test = df_test.values
    print("\nGenerating predictions on test set...")

    # 1. XGBoost Prediction (using the pipeline trained on CV folds)
    print("Training final XGBoost model on all data for test prediction...")
    xgb_pipeline.fit(X, y)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    print(f"XGBoost predictions generated for {len(y_pred_xgb)} samples.")

    # 2. CNN Prediction
    X_test_cnn = X_test / 255.0
    y_pred_cnn_probs = cnn_model.predict(X_test_cnn, verbose=0)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
    print(f"CNN predictions generated for {len(y_pred_cnn)} samples.")
```