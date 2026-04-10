import matplotlib.pyplot as plt

def plot_data(df):
    print("\n --- Data Visualization ---")

    df.hist(figsize=(8,6))
    plt.tight_layout()
    plt.show()

    plt.scatter(df['hours_studied'],df['final_score'])
    plt.xlabel('Hours Studied')
    plt.ylabel('Final Score')
    plt.title("Hours vs Score")
    plt.show()

    plt.scatter(df['previous_score'],df['final_score'])
    plt.xlabel("Previous Score")
    plt.ylabel("Final Score")
    plt.title("Previous Score vs Final Score")
    plt.show()