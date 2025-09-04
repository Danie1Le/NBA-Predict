import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

class NBAPredictorTF:
    """
    TensorFlow/Keras implementation for NBA game outcome prediction.
    Provides multiple model architectures for comparison.
    """
    
    def __init__(self, input_size, model_type='dense'):
        self.input_size = input_size
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
    def build_dense_model(self, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        """
        Build a dense neural network model.
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(self.input_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model
    
    def build_lstm_model(self, lstm_units=64, dense_units=[64, 32], dropout_rate=0.3):
        """
        Build an LSTM model for time series prediction.
        """
        model = keras.Sequential()
        
        # LSTM layer
        model.add(layers.LSTM(lstm_units, return_sequences=True, input_shape=(1, self.input_size)))
        model.add(layers.Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(layers.LSTM(lstm_units // 2, return_sequences=False))
        model.add(layers.Dropout(dropout_rate))
        
        # Dense layers
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model
    
    def build_hybrid_model(self, dense_units=128, lstm_units=64, dropout_rate=0.3):
        """
        Build a hybrid model combining dense and LSTM layers.
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_size,))
        
        # Dense branch
        dense_branch = layers.Dense(dense_units, activation='relu')(inputs)
        dense_branch = layers.BatchNormalization()(dense_branch)
        dense_branch = layers.Dropout(dropout_rate)(dense_branch)
        dense_branch = layers.Dense(dense_units // 2, activation='relu')(dense_branch)
        dense_branch = layers.BatchNormalization()(dense_branch)
        dense_branch = layers.Dropout(dropout_rate)(dense_branch)
        
        # Reshape for LSTM
        lstm_input = layers.Reshape((1, dense_units // 2))(dense_branch)
        
        # LSTM branch
        lstm_branch = layers.LSTM(lstm_units, return_sequences=False)(lstm_input)
        lstm_branch = layers.Dropout(dropout_rate)(lstm_branch)
        
        # Final dense layers
        combined = layers.Dense(32, activation='relu')(lstm_branch)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(dropout_rate)(combined)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(combined)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_attention_model(self, hidden_units=128, attention_heads=8, dropout_rate=0.3):
        """
        Build a model with attention mechanism for feature importance.
        """
        inputs = layers.Input(shape=(self.input_size,))
        
        # Feature embedding
        x = layers.Dense(hidden_units, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=attention_heads, 
            key_dim=hidden_units // attention_heads
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff = layers.Dense(hidden_units * 2, activation='relu')(x)
        ff = layers.Dropout(dropout_rate)(ff)
        ff = layers.Dense(hidden_units, activation='relu')(ff)
        
        # Add & Norm
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_model(self, **kwargs):
        """
        Build the specified model type.
        """
        if self.model_type == 'dense':
            self.model = self.build_dense_model(**kwargs)
        elif self.model_type == 'lstm':
            self.model = self.build_lstm_model(**kwargs)
        elif self.model_type == 'hybrid':
            self.model = self.build_hybrid_model(**kwargs)
        elif self.model_type == 'attention':
            self.model = self.build_attention_model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model with specified optimizer and loss function.
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'adamw':
            opt = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, 
              early_stopping=True, reduce_lr=True, verbose=1):
        """
        Train the model with the given data.
        """
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Prepare data for LSTM if needed
        if self.model_type == 'lstm':
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        
        # Callbacks
        callbacks = []
        
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        if reduce_lr:
            reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
            )
            callbacks.append(reduce_lr_cb)
        
        # Train model
        print(f"Training {self.model_type} TensorFlow model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_accuracy, test_auc = self.model.evaluate(
            X_test_scaled, y_test, verbose=0
        )
        
        print(f'Test Accuracy: {test_accuracy:.4f}')
        print(f'Test AUC: {test_auc:.4f}')
        
        return history, (X_test_scaled, y_test)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'lstm':
            X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        predictions = self.model.predict(X_scaled)
        binary_predictions = (predictions > 0.5).astype(int)
        
        return binary_predictions, predictions
    
    def get_feature_importance(self, X_sample, feature_names):
        """
        Get feature importance using gradient-based methods.
        """
        X_scaled = self.scaler.transform(X_sample)
        
        if self.model_type == 'lstm':
            X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(tf.Variable(X_scaled))
            predictions = self.model(X_scaled)
        
        gradients = tape.gradient(predictions, tf.Variable(X_scaled))
        importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
        
        if self.model_type == 'lstm':
            importance = importance[0]  # Remove sequence dimension
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance))
        return importance_dict

def train_tensorflow_model(X, y, model_type='dense', epochs=100, batch_size=32, learning_rate=0.001):
    """
    Convenience function to train a TensorFlow model.
    """
    predictor = NBAPredictorTF(X.shape[1], model_type=model_type)
    predictor.build_model()
    predictor.compile_model(learning_rate=learning_rate)
    
    history, test_data = predictor.train(
        X, y, epochs=epochs, batch_size=batch_size
    )
    
    return predictor, test_data, history

def compare_tensorflow_models(X, y, model_types=['dense', 'hybrid', 'attention'], epochs=50):
    """
    Compare different TensorFlow model architectures.
    """
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        predictor, test_data, history = train_tensorflow_model(
            X, y, model_type=model_type, epochs=epochs
        )
        
        # Make predictions
        X_test, y_test = test_data
        y_pred, y_proba = predictor.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
        
        results[model_type] = {
            'model': predictor,
            'accuracy': accuracy,
            'auc': auc,
            'history': history
        }
        
        print(f"{model_type.upper()} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
    
    return results
