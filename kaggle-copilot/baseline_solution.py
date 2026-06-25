import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Set device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 0. FEATURE ENGINEERING ---

def extract_derived_image_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generates domain-specific summary features from the 784 pixel columns (0-255 scale).
    Returns a DataFrame containing only the newly engineered features.
    """
    pixel_cols = [f'pixel{i}' for i in range(784)]
    
    # 1. Total Ink Density (Mean intensity)
    ink_density = X[pixel_cols].mean(axis=1).rename('FE_Ink_Density')
    
    # Vertical division points
    top_half_cols = [f'pixel{i}' for i in range(28 * 14)]
    bottom_half_cols = [f'pixel{i}' for i in range(28 * 14, 784)]
    
    # Horizontal division points (i % 28 is column index)
    left_cols = []
    right_cols = []
    for i in range(784):
        if i % 28 < 14: 
            left_cols.append(f'pixel{i}')
        else:           
            right_cols.append(f'pixel{i}')
            
    # 2. Vertical Imbalance (Top - Bottom)
    top_density = X[top_half_cols].mean(axis=1)
    bottom_density = X[bottom_half_cols].mean(axis=1)
    vertical_imbalance = (top_density - bottom_density).rename('FE_Vertical_Imbalance')
    
    # 3. Horizontal Imbalance (Left - Right)
    left_density = X[left_cols].mean(axis=1)
    right_density = X[right_cols].mean(axis=1)
    horizontal_imbalance = (left_density - right_density).rename('FE_Horizontal_Imbalance')

    # 4. Center Density (14x14 center box: Rows 7-20, Columns 7-20)
    center_cols = []
    for r in range(7, 21): 
        for c in range(7, 21): 
            idx = r * 28 + c
            center_cols.append(f'pixel{idx}')
                
    center_density = X[center_cols].mean(axis=1).rename('FE_Center_Density')
    
    X_derived_features = pd.concat([
        ink_density, vertical_imbalance, horizontal_imbalance, center_density
    ], axis=1)
    
    return X_derived_features

# --- 1. DATA PREPARATION UTILITIES ---

def augment_for_lgbm(X_raw: pd.DataFrame):
    """Calculates features, normalizes pixels, and concatenates them for LGBM input."""
    # 1. Normalize raw pixel data (784 features)
    X_pixels_normalized = X_raw / 255.0
    
    # 2. Extract engineered features (based on 0-255 scale)
    X_engineered = extract_derived_image_features(X_raw)
    
    # 3. Concatenate normalized pixels and engineered features
    X_lgbm = pd.concat([X_pixels_normalized, X_engineered], axis=1)
    return X_lgbm.values

class DigitDataset(Dataset):
    def __init__(self, X_raw, y=None, transform=None):
        if hasattr(X_raw, 'values'):
            X_raw = X_raw.values
        
        # Normalize pixels to [0, 1] and reshape to (C, H, W) for CNN
        self.X = torch.tensor(X_raw / 255.0, dtype=torch.float32).view(-1, 1, 28, 28).clone().detach()
        
        if y is not None:
            if hasattr(y, 'values'):
                y = y.values
            self.y = torch.tensor(y, dtype=torch.long).clone().detach()
        else:
            self.y = None
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# --- 2. LIGHTGBM MODEL ---

def train_lgbm_model(X_raw, y):
    X_train_lgbm = augment_for_lgbm(X_raw)
    
    print("Training LightGBM model...")
    lgbm_model = lgb.LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_model.fit(X_train_lgbm, y)
    return lgbm_model

# --- 3. CNN MODEL (PyTorch) ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Standard CNN architecture for 28x28 grayscale image
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

def train_cnn_model(X_raw, y, epochs=10, batch_size=128):
    # Increased epochs to 10 for better training on the full dataset
    train_dataset = DigitDataset(X_raw, y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    cnn_model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters())
    
    print(f"Training PyTorch CNN model on {DEVICE} ({epochs} epochs)...")
    cnn_model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

        epoch_loss = total_loss / total_samples
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
            
    return cnn_model


def predict_cnn_probs(cnn_model, X_raw, batch_size=128):
    test_dataset = DigitDataset(X_raw)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    cnn_model.eval()
    all_probs = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, list):
                images = images[0]
                
            images = images.to(DEVICE)
            outputs = cnn_model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs, axis=0)

# --- 4. ENSEMBLE PREDICTION ---

def ensemble_predict(X_test_raw, lgbm_model, cnn_model):
    
    # Predict probabilities from LGBM using augmented data (788 features)
    X_test_lgbm = augment_for_lgbm(X_test_raw)
    lgbm_probs = lgbm_model.predict_proba(X_test_lgbm)
    
    # Predict probabilities from CNN (784 features)
    cnn_probs = predict_cnn_probs(cnn_model, X_test_raw)

    print("Generating predictions via weighted soft voting...")
    
    # Weighted Soft Voting: CNN gets higher weight (W_CNN=0.7, W_LGBM=0.3)
    W_LGBM = 0.3
    W_CNN = 0.7
    
    avg_probs = (W_LGBM * lgbm_probs) + (W_CNN * cnn_probs)
    
    predictions = np.argmax(avg_probs, axis=1)
    
    return predictions

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    # Define file paths
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("Error: train.csv and test.csv must be present in the current directory.")
    else:
        print("Loading data...")
        # Load Training Data (X_train_raw = features 0-255, y_train = label)
        df_train = pd.read_csv(TRAIN_FILE)
        X_train_raw = df_train.drop('label', axis=1)
        y_train = df_train['label']
        
        # Load Test Data
        X_test_raw = pd.read_csv(TEST_FILE)
        
        print(f"X_train shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
        
        # 5. Model Training (Training on full dataset for final submission)
        
        # Train LGBM (uses 784 pixels + 4 FE features = 788 features)
        final_lgbm_model = train_lgbm_model(X_train_raw, y_train)
        
        # Train CNN (uses only 784 pixels) with increased epochs
        final_cnn_model = train_cnn_model(X_train_raw, y_train, epochs=10)
        
        # 6. Prediction
        final_predictions = ensemble_predict(X_test_raw, final_lgbm_model, final_cnn_model)
        
        # 7. Submission File Generation
        
        submission_df = pd.DataFrame({
            'ImageId': range(1, len(final_predictions) + 1),
            'Label': final_predictions
        })
        
        submission_file = 'digit_recognizer_ensemble_pytorch_fe_submission.csv'
        submission_df.to_csv(submission_file, index=False)
        print(f"Submission file saved successfully as {submission_file}")