def clean_data(df):
    print("Before cleaning:", len(df))
    print("\n --- Cleaning data ---")

    df = df.dropna()
    df = df.drop_duplicates()


    df = df[df['hours_studied'] >= 0]
    df = df[df['attendance'] >= 0]
    df = df[df['sleep_hours'] >= 0]
    print("Cleaning done ✅")
    print("After cleaning:", len(df))
    return df