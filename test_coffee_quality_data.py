import pytest
import pandas as pd
import numpy as np
from Coffee_Data import (
    load_data,
    filter_above_avg_score,
    filter_arabica,
    filter_robusta,
    sweetness_ML,
)


# Test 1: Data Loading
def test_data_load():
    df = load_data("merged_data_cleaned.csv")

    assert df is not None
    assert len(df) > 0
    assert len(df.columns) == 44
    assert "Total.Cup.Points" in df.columns
    assert "Species" in df.columns
    assert "Country.of.Origin" in df.columns


# Edge Case: Missing Data
def test_data_missing():
    with pytest.raises(FileNotFoundError):
        load_data("data_does_not_exist.csv")


# Test 2: Data Transformation
def test_filter_above_avg_score():
    df = load_data("merged_data_cleaned.csv")
    result = filter_above_avg_score(df)

    assert len(result) > 0
    assert (result["Total.Cup.Points"] > df["Total.Cup.Points"].mean()).all()


# Test 3: Data Filtering (Arabica Coffee Type)
def test_filter_arabica():
    df = load_data("merged_data_cleaned.csv")
    result = filter_arabica(df)
    assert len(result) > 0
    assert (result["Species"] == "Arabica").all()


# Test 3: Data Filtering (Robusta Coffee Type)
def test_filter_robusta():
    df = load_data("merged_data_cleaned.csv")
    result = filter_robusta(df)
    assert len(result) > 0
    assert (result["Species"] == "Robusta").all()


# Test 4: Machine Learning Model Functionality
def test_sweetness_prediction():
    df = load_data("merged_data_cleaned.csv")

    model, predictions = sweetness_ML(df)

    assert model is not None
    assert len(predictions) == len(df)
    assert len(predictions) > 0
    assert np.isfinite(predictions).all()


# End_to_End Test
def test_end_to_end():
    df = load_data("merged_data_cleaned.csv")

    above_average = filter_above_avg_score(df)

    arabica = filter_arabica(df)

    robusta = filter_robusta(df)

    model, predictions = sweetness_ML(df)

    assert len(df) > 0
    assert len(above_average) > 0
    assert len(arabica) > 0
    assert len(robusta) > 0
    assert len(predictions) == len(df)
    assert np.isfinite(predictions).all()
