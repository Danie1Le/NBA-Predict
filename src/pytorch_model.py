import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

class NBAPredictorNet(nn.Module):
    """
    PyTorch neural network for NBA game outcome prediction.
    Combines feedforward layers with LSTM for time series features.
    """
    
    def __init__(self, input_size, hidden_size=128, lstm_hidden=64, num_layers=2, dropout=0.3):
        super(NBAPredictorNet, self).__init__()
        
        # Feedforward layers for static features
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for time series features (if we have sequential data)
        self.lstm = nn.LSTM(
            input_size=hidden_size // 4,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feedforward processing
        x = self.feedforward(x)
        
        # Reshape for LSTM (batch_size, sequence_length, features)
        # For now, we'll treat each sample as a single timestep
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Final classification
        output = self.classifier(lstm_out)
        return output

class NBAPredictorLSTM(nn.Module):
    """
    Alternative PyTorch model with pure LSTM architecture for time series prediction.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(NBAPredictorLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Reshape input for LSTM
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(lstm_out)
        return output

def prepare_pytorch_data(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for PyTorch training with proper scaling and tensor conversion.
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
    
    return (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler)

def train_pytorch_model(X, y, model_type='hybrid', epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train a PyTorch model for NBA game prediction.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_type: 'hybrid' (feedforward + LSTM) or 'lstm' (pure LSTM)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        trained_model, test_data, scaler
    """
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_pytorch_data(X, y)
    
    # Create model
    input_size = X_train.shape[1]
    if model_type == 'hybrid':
        model = NBAPredictorNet(input_size)
    elif model_type == 'lstm':
        model = NBAPredictorLSTM(input_size)
    else:
        raise ValueError("model_type must be 'hybrid' or 'lstm'")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    train_losses = []
    
    print(f"Training {model_type} PyTorch model...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = (test_outputs > 0.5).float()
        test_probabilities = test_outputs.numpy()
        test_predictions_np = test_predictions.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test.numpy(), test_predictions_np)
    try:
        auc = roc_auc_score(y_test.numpy(), test_probabilities)
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test ROC AUC: {auc:.4f}')
    except:
        print(f'Test Accuracy: {accuracy:.4f}')
    
    return model, (X_test, y_test), scaler, train_losses

def predict_pytorch(model, X, scaler):
    """
    Make predictions using a trained PyTorch model.
    """
    model.eval()
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        predictions = model(X_tensor)
        probabilities = predictions.numpy()
        binary_predictions = (predictions > 0.5).float().numpy()
    
    return binary_predictions, probabilities

def create_team_sequence_data(df, features, sequence_length=5):
    """
    Create sequential data for LSTM training by grouping games by team.
    This creates time series data where each sequence represents a team's recent games.
    """
    # Sort by team and date
    df_sorted = df.sort_values(['Team_ID', 'GAME_DATE_REAL'])
    
    sequences = []
    targets = []
    
    for team_id in df_sorted['Team_ID'].unique():
        team_data = df_sorted[df_sorted['Team_ID'] == team_id]
        
        # Create sequences of specified length
        for i in range(sequence_length, len(team_data)):
            sequence = team_data.iloc[i-sequence_length:i][features].values
            target = (team_data.iloc[i]['WL'] == 'W').astype(int)
            
            sequences.append(sequence)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)
