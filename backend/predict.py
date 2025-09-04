def predict_outcome(model, X):
    """
    Use the trained model to predict the outcome of games.
    Returns predicted labels (1=win, 0=loss).
    """
    return model.predict(X) 