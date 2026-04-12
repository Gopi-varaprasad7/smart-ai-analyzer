from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_test, y_test):
    print("\n --- Evaluating Model ---")

    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test,predictions)

    print("Mean absolute error:",error)
    return error