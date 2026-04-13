import numpy as np

def train_from_scratch(df, lr=0.0001, epochs=1000):
    print("\n--- Training From Scratch ---")

    # Features and target
    X = df[['hours_studied', 'attendance', 'previous_score', 'sleep_hours']].values
    y = df['final_score'].values

    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    n = len(y)

    for epoch in range(epochs):
        # Predictions
        y_pred = np.dot(X, weights) + bias

        # Error
        error = y_pred - y

        # Gradients
        dw = (1/n) * np.dot(X.T, error)
        db = (1/n) * np.sum(error)

        # Update weights
        weights -= lr * dw
        bias -= lr * db

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            loss = np.mean(error**2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Training complete ✅")
    print("Weights:", weights)
    print("Bias:", bias)

    return weights, bias