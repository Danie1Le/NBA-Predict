import React from 'react';

export const MODEL_NAMES = {
  xgb: 'XGBoost',
  rf: 'Random forest',
  logreg: 'Logistic regression',
  ensemble: 'Ensemble',
  pytorch: 'PyTorch network',
  tensorflow: 'TensorFlow network',
};

const MODEL_DESCRIPTIONS = {
  xgb: 'Gradient-boosted decision trees.',
  rf: 'Many decision trees voting together.',
  logreg: 'A simple linear model, easy to interpret.',
  ensemble: 'Averages every available model.',
  pytorch: 'A PyTorch neural network.',
  tensorflow: 'A TensorFlow neural network.',
};

// Segmented control built on native radios: arrow keys move between models.
const ModelSelector = ({ models, selectedModel, recommendedModel, onModelSelect }) => {
  const description = MODEL_DESCRIPTIONS[selectedModel] ?? '';

  return (
    <fieldset aria-describedby="model-description">
      <legend className="sr-only">Model</legend>
      <div className="mb-1.5 flex flex-wrap items-baseline gap-x-2 text-sm">
        <span aria-hidden="true" className="font-medium text-label-2">Model</span>
        <p id="model-description" className="text-label-2">
          {description}
          {selectedModel === recommendedModel && ' Recommended.'}
        </p>
      </div>
      <div className="grid grid-cols-3 gap-1 rounded-xl bg-fill p-1">
        {models.map((model) => (
          <label key={model} className="relative min-w-0 cursor-pointer">
            <input
              type="radio"
              name="model"
              value={model}
              checked={selectedModel === model}
              onChange={() => onModelSelect(model)}
              className="peer sr-only"
            />
            <span
              className="flex h-full min-h-[44px] items-center justify-center rounded-lg px-2 py-1.5 text-center text-[0.9375rem] font-medium leading-tight
                         text-label-2 transition-colors hover:text-label
                         peer-checked:bg-raised peer-checked:text-label peer-checked:shadow-sm
                         peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-tint"
            >
              {MODEL_NAMES[model] ?? model}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
};

export default ModelSelector;
