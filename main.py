import pandas as pd
from src.data_cleaning import clean_data
from src.visualization import plot_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.model_from_scratch import train_from_scratch

df = pd.read_csv('data/student_data.csv');
df = clean_data(df)

print("\nCleaned Data:\n", df.head())
# print("First 5 rows \n",df.head())
# print("\n Info: \n")
# print(df.info())
# print("\n Summary: \n", df.describe())
# print("\n Missing values: \n",df.isnull().sum())
# print(df.corr())
plot_data(df)
model, X_test, y_test = train_model(df)
error = evaluate_model(model ,X_test , y_test)
weights, bias = train_from_scratch(df)