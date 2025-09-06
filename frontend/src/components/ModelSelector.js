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
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-2">
        {models.map((model) => (
          <button
            key={model}
            onClick={() => onModelSelect(model)}
            disabled={disabled}
            className={`p-3 rounded-lg border-2 transition-all duration-200 text-left
              ${selectedModel === model
                ? 'border-nba-orange bg-nba-orange/20 text-white'
                : 'border-white/20 bg-white/5 text-gray-300 hover:border-white/40 hover:bg-white/10'
              }
              disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <div className="flex items-center space-x-2">
              <div className={`p-1.5 rounded ${
                selectedModel === model ? 'bg-nba-orange/30' : 'bg-white/10'
              }`}>
                {modelIcons[model]}
              </div>
              
              <div className="flex-1 min-w-0">
                <h3 className="font-medium text-white text-sm truncate">
                  {modelNames[model]}
                </h3>
                <p className="text-xs text-gray-400 truncate">
                  {modelDescriptions[model]}
                </p>
              </div>
              
              {selectedModel === model && (
                <div className="w-2 h-2 bg-nba-orange rounded-full flex-shrink-0"></div>
              )}
            </div>
          </button>
        ))}
      </div>
      
      <div className="bg-blue-500/20 border border-blue-500/50 rounded-lg p-3">
        <div className="flex items-center space-x-2 text-blue-200">
          <Sparkles className="h-3 w-3" />
          <span className="text-xs font-medium">Recommended: XGBoost</span>
        </div>
        <p className="text-xs text-blue-300 mt-1">
          Best performance for predictions
        </p>
      </div>
    </div>
  );
};

export default ModelSelector;
