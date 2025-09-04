import sys
import os

# Add the parent directory to the Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import load_and_clean_data
from feature_engineering import create_features
from train_model import train_model
from predict import predict_outcome
from pytorch_model import train_pytorch_model, predict_pytorch
from tensorflow_model import train_tensorflow_model, compare_tensorflow_models
from ensemble_model import train_ensemble_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_comprehensive_comparison():
    """
    Run a comprehensive comparison of all model types: traditional ML, PyTorch, TensorFlow, and ensemble.
    """
    print("🏀 NBA Game Prediction - Deep Learning vs Traditional ML Comparison")
    print("="*80)
    
    # 1. Load and clean data
    print("\n1. Loading and preprocessing data...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'NBA_GAMES.csv')
    df = load_and_clean_data(data_path)
    
    # 2. Feature engineering
    print("\n2. Creating features...")
    df = create_features(df)
    
    # 3. Select features and target
    features = [
        'HOME',
        # Team rolling stats
        'PTS_rolling5', 'FG_PCT_rolling5', 'FG3_PCT_rolling5', 'FT_PCT_rolling5',
        'REB_rolling5', 'AST_rolling5', 'TOV_rolling5',
        'PTS_rolling10', 'FG_PCT_rolling10', 'FG3_PCT_rolling10', 'FT_PCT_rolling10',
        'REB_rolling10', 'AST_rolling10', 'TOV_rolling10',
        'WIN_STREAK5',
        # Team season strength
        'SEASON_WIN_PCT',
        # Opponent rolling stats
        'OPP_PTS_rolling5', 'OPP_FG_PCT_rolling5', 'OPP_FG3_PCT_rolling5', 'OPP_FT_PCT_rolling5',
        'OPP_REB_rolling5', 'OPP_AST_rolling5', 'OPP_TOV_rolling5',
        'OPP_PTS_rolling10', 'OPP_FG_PCT_rolling10', 'OPP_FG3_PCT_rolling10', 'OPP_FT_PCT_rolling10',
        'OPP_REB_rolling10', 'OPP_AST_rolling10', 'OPP_TOV_rolling10',
        # Opponent season strength
        'OPP_SEASON_WIN_PCT',
        # Rest days
        'REST_DAYS', 'OPP_REST_DAYS'
    ]
    
    X = df[features]
    y = (df['WL'] == 'W').astype(int)  # 1 for win, 0 for loss
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(features)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Results storage
    results = {}
    
    # 4. Traditional ML Models
    print("\n" + "="*60)
    print("3. TRADITIONAL MACHINE LEARNING MODELS")
    print("="*60)
    
    traditional_models = ['xgb', 'rf', 'logreg']
    for model_type in traditional_models:
        print(f"\nTraining {model_type.upper()}...")
        try:
            model, X_test, y_test = train_model(X, y, model_type=model_type)
            y_pred = predict_outcome(model, X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.0
            
            results[f'traditional_{model_type}'] = {
                'accuracy': accuracy,
                'auc': auc,
                'model': model
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    # 5. PyTorch Models
    print("\n" + "="*60)
    print("4. PYTORCH DEEP LEARNING MODELS")
    print("="*60)
    
    pytorch_models = ['hybrid', 'lstm']
    for model_type in pytorch_models:
        print(f"\nTraining PyTorch {model_type.upper()}...")
        try:
            model, test_data, scaler, train_losses = train_pytorch_model(
                X, y, model_type=model_type, epochs=50
            )
            X_test, y_test = test_data
            
            y_pred, y_proba = predict_pytorch(model, X_test, scaler)
            accuracy = accuracy_score(y_test.numpy(), y_pred)
            try:
                auc = roc_auc_score(y_test.numpy(), y_proba)
            except:
                auc = 0.0
            
            results[f'pytorch_{model_type}'] = {
                'accuracy': accuracy,
                'auc': auc,
                'model': model,
                'scaler': scaler,
                'train_losses': train_losses
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    # 6. TensorFlow Models
    print("\n" + "="*60)
    print("5. TENSORFLOW DEEP LEARNING MODELS")
    print("="*60)
    
    try:
        tf_results = compare_tensorflow_models(X, y, model_types=['dense', 'hybrid', 'attention'], epochs=30)
        for model_type, result in tf_results.items():
            results[f'tensorflow_{model_type}'] = {
                'accuracy': result['accuracy'],
                'auc': result['auc'],
                'model': result['model'],
                'history': result['history']
            }
    except Exception as e:
        print(f"TensorFlow models failed: {e}")
    
    # 7. Ensemble Model
    print("\n" + "="*60)
    print("6. ENSEMBLE MODEL (ALL APPROACHES COMBINED)")
    print("="*60)
    
    try:
        ensemble, ensemble_results = train_ensemble_model(
            X, y, 
            use_pytorch=True, 
            use_tensorflow=True, 
            use_traditional=True
        )
        results['ensemble'] = ensemble_results
    except Exception as e:
        print(f"Ensemble model failed: {e}")
    
    # 8. Results Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': result['accuracy'],
            'AUC': result['auc'],
            'Type': name.split('_')[0]
        }
        for name, result in results.items()
        if 'accuracy' in result
    ])
    
    if not results_df.empty:
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print(results_df.to_string(index=False))
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Accuracy comparison
        plt.subplot(2, 2, 1)
        sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Type')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # AUC comparison
        plt.subplot(2, 2, 2)
        sns.barplot(data=results_df, x='Model', y='AUC', hue='Type')
        plt.title('Model AUC Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Accuracy vs AUC scatter
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=results_df, x='Accuracy', y='AUC', hue='Type', s=100)
        plt.title('Accuracy vs AUC')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Model type performance
        plt.subplot(2, 2, 4)
        type_performance = results_df.groupby('Type')[['Accuracy', 'AUC']].mean()
        type_performance.plot(kind='bar', ax=plt.gca())
        plt.title('Average Performance by Model Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.tight_layout()
        plt.show()
        
        # Best model recommendation
        best_model = results_df.iloc[0]
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
        print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        print(f"   AUC: {best_model['AUC']:.4f}")
        print(f"   Type: {best_model['Type']}")
    
    return results, results_df

def run_quick_demo():
    """
    Run a quick demonstration of the deep learning capabilities.
    """
    print("🏀 NBA Prediction - Quick Deep Learning Demo")
    print("="*50)
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'NBA_GAMES.csv')
    df = load_and_clean_data(data_path)
    df = create_features(df)
    
    # Select features
    features = [
        'HOME', 'PTS_rolling5', 'FG_PCT_rolling5', 'REB_rolling5', 'AST_rolling5',
        'SEASON_WIN_PCT', 'OPP_PTS_rolling5', 'OPP_FG_PCT_rolling5', 'OPP_SEASON_WIN_PCT',
        'REST_DAYS', 'OPP_REST_DAYS'
    ]
    
    X = df[features]
    y = (df['WL'] == 'W').astype(int)
    
    print(f"Training on {len(X)} games with {len(features)} features")
    
    # Quick comparison
    models_to_test = [
        ('XGBoost', lambda: train_model(X, y, model_type='xgb')),
        ('PyTorch Hybrid', lambda: train_pytorch_model(X, y, model_type='hybrid', epochs=20)),
        ('TensorFlow Dense', lambda: train_tensorflow_model(X, y, model_type='dense', epochs=20))
    ]
    
    results = {}
    for name, train_func in models_to_test:
        print(f"\nTraining {name}...")
        try:
            if 'PyTorch' in name:
                model, test_data, scaler, _ = train_func()
                X_test, y_test = test_data
                y_pred, y_proba = predict_pytorch(model, X_test, scaler)
                accuracy = accuracy_score(y_test.numpy(), y_pred)
            elif 'TensorFlow' in name:
                model, test_data, _ = train_func()
                X_test, y_test = test_data
                y_pred, y_proba = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
            else:
                model, X_test, y_test = train_func()
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = accuracy
            print(f"  Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    if results:
        best = max(results, key=results.get)
        print(f"\n🏆 Best model: {best} ({results[best]:.4f})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Prediction Deep Learning Demo')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='Run full comparison or quick demo')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_comprehensive_comparison()
    else:
        run_quick_demo()
