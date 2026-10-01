import pytest
import pandas as pd
import numpy as np

from Coffee_Data import (
    load_data,
    filter_above_avg_score,
    filter_arabica,
    filter_robusta,
    sweetness_ml,
)

data_file = "merged_data_cleaned.csv"


# Test 1: Data Loading
def test_data_load():
    df = load_data(data_file)

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
    df = load_data(data_file)
    result = filter_above_avg_score(df)

    assert len(result) > 0
    assert (result["Total.Cup.Points"] > df["Total.Cup.Points"].mean()).all()


# Edge Case: Empty Dataframe
def test_filter_above_avg_score_empty_dataframe():
    df = pd.DataFrame({"Total.Cup.Points": []})
    result = filter_above_avg_score(df)

    assert result.empty


# Test 3: Data Filtering (Arabica Coffee Type)
def test_filter_arabica():
    df = load_data(data_file)
    result = filter_arabica(df)
    assert len(result) > 0
    assert (result["Species"] == "Arabica").all()


# Test 3: Data Filtering (Robusta Coffee Type)
def test_filter_robusta():
    df = load_data(data_file)
    result = filter_robusta(df)
    assert len(result) > 0
    assert (result["Species"] == "Robusta").all()


# Edge Case: No matching coffee type
def test_filter_arabica_no_match():
    df = pd.DataFrame(
        {
            "Species": ["Robusta", "Robusta"],
            "Total.Cup.Points": [80, 82],
        }
    )
    result = filter_arabica(df)
    assert result.empty


# Test 4: Machine Learning Model Functionality
def test_sweetness_prediction():
    df = load_data(data_file)

    model, predictions = sweetness_ml(df)

    assert model is not None
    assert len(predictions) == len(df)
    assert len(predictions) > 0
    assert np.isfinite(predictions).all()


# Edge Case: Small Validation Dataset for ML
def test_sweetness_prediction_small_dataset():
    df = pd.DataFrame(
        {"Sweetness": [8.0, 9.0, 10.0], "Total.Cup.Points": [80.0, 82.0, 84.0]}
    )

    model, predictions = sweetness_ml(df)
    assert len(predictions) == 3
    assert np.isfinite(predictions).all()


# End_to_End Test
def test_end_to_end():
    df = load_data(data_file)

    above_average = filter_above_avg_score(df)

    arabica = filter_arabica(df)

    robusta = filter_robusta(df)

    model, predictions = sweetness_ml(df)

    assert len(df) > 0
    assert len(above_average) > 0
    assert len(arabica) > 0
    assert len(robusta) > 0
    assert len(predictions) == len(df)
    assert np.isfinite(predictions).all()
