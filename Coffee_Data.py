import pandas as pd
from sklearn.linear_model import LinearRegression

def load_data(filepath):
    return pd.read_csv(filepath)


def filter_above_avg_score(df):
    average_quality = df["Total.Cup.Points"].mean()
    return df[df["Total.Cup.Points"] > average_quality]


def filter_arabica(df):
    return df[df["Species"] == "Arabica"]


def filter_robusta(df):
    return df[df["Species"] == "Robusta"]


def sweetness_ML(df):
    x = df[["Sweetness"]]
    y = df["Total.Cup.Points"]
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    return model, predictions
