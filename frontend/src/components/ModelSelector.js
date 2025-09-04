import { Brain, Cpu, Layers, Sparkles, Target, Zap } from 'lucide-react';
import React from 'react';

const ModelSelector = ({ models, selectedModel, onModelSelect, disabled }) => {
  const modelIcons = {
    'xgb': <Zap className="h-5 w-5" />,
    'rf': <Layers className="h-5 w-5" />,
    'logreg': <Target className="h-5 w-5" />,
    'pytorch': <Brain className="h-5 w-5" />,
    'tensorflow': <Cpu className="h-5 w-5" />,
    'ensemble': <Sparkles className="h-5 w-5" />
  };

  const modelNames = {
    'xgb': 'XGBoost',
    'rf': 'Random Forest',
    'logreg': 'Logistic Regression',
    'pytorch': 'PyTorch Neural Network',
    'tensorflow': 'TensorFlow/Keras',
    'ensemble': 'Ensemble (All Models)'
  };

  const modelDescriptions = {
    'xgb': 'Gradient boosting with excellent performance',
    'rf': 'Ensemble of decision trees',
    'logreg': 'Linear model with good interpretability',
    'pytorch': 'Deep learning with neural networks',
    'tensorflow': 'Advanced deep learning framework',
    'ensemble': 'Combines all models for best accuracy'
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3">
        {models.map((model) => (
          <button
            key={model}
            onClick={() => onModelSelect(model)}
            disabled={disabled}
            className={`p-4 rounded-lg border-2 transition-all duration-200 text-left
              ${selectedModel === model
                ? 'border-nba-orange bg-nba-orange/20 text-white'
                : 'border-white/20 bg-white/5 text-gray-300 hover:border-white/40 hover:bg-white/10'
              }
              disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <div className="flex items-center space-x-3">
              <div className={`p-2 rounded-lg ${
                selectedModel === model ? 'bg-nba-orange/30' : 'bg-white/10'
              }`}>
                {modelIcons[model]}
              </div>
              
              <div className="flex-1">
                <h3 className="font-semibold text-white">
                  {modelNames[model]}
                </h3>
                <p className="text-sm text-gray-400">
                  {modelDescriptions[model]}
                </p>
              </div>
              
              {selectedModel === model && (
                <div className="w-3 h-3 bg-nba-orange rounded-full"></div>
              )}
            </div>
          </button>
        ))}
      </div>
      
      <div className="bg-blue-500/20 border border-blue-500/50 rounded-lg p-4">
        <div className="flex items-center space-x-2 text-blue-200">
          <Sparkles className="h-4 w-4" />
          <span className="text-sm font-medium">Recommended: Ensemble Model</span>
        </div>
        <p className="text-xs text-blue-300 mt-1">
          Combines all models for the most accurate predictions
        </p>
      </div>
    </div>
  );
};

export default ModelSelector;
