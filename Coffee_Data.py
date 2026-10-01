import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(filepath):
    """Read the coffee quality dataset from the CSV file."""
    return pd.read_csv(filepath)


def filter_above_avg_score(df):
    """Return coffee cups with scores above the dataset quality average."""
    average_quality = df["Total.Cup.Points"].mean()
    return df[df["Total.Cup.Points"] > average_quality]


def filter_arabica(df):
    """Return coffee cups classified as Arabica type."""
    return df[df["Species"] == "Arabica"]


def filter_robusta(df):
    """Return coffee cups classified as Robusta type."""
    return df[df["Species"] == "Robusta"]


def sweetness_ml(df):
    """Fit linear regression model using sweetness to
    predict coffee quality."""
    x = df[["Sweetness"]]
    y = df["Total.Cup.Points"]
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    return model, predictions
