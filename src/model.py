from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def train_model(df):
    print("\n--- Training Model ---")

    X = df[['hours_studied','attendance','previous_score','sleep_hours']]
    y = df['final_score']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state =42)
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model,X_test,y_test