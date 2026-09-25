import pandas as pd
import pytest

from datadoctor import Doctor


@pytest.fixture
def sample_df():
    data = {
        "age": [25, 30, 35, None, 40, 30],
        "salary": [50000, 60000, 70000, 80000, None, 60000],
        "department": ["HR", "Tech", "Tech", "Finance", "HR", "Tech"],
    }
    return pd.DataFrame(data)


def test_diagnose(sample_df):
    doc = Doctor(sample_df)
    report = doc.diagnose()
    assert report["num_rows"] == 6
    assert report["missing_values"]["age"] == 1
    assert report["missing_values"]["salary"] == 1
    assert report["duplicate_rows"] == 1


def test_treat_removes_missing_and_duplicates(sample_df):
    doc = Doctor(sample_df)
    cleaned = doc.treat()
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) == 5


def test_convert_types_numeric_strings():
    df = pd.DataFrame(
        {
            "age": ["25", "30", "35"],
            "name": ["alice", "bob", "carol"],
        }
    )
    doc = Doctor(df)
    result = doc.convert_types()
    assert pd.api.types.is_numeric_dtype(result["age"])
    assert not pd.api.types.is_numeric_dtype(result["name"])


def test_convert_types_dates():
    df = pd.DataFrame({"joined": ["2021-01-01", "2021-06-15", "2022-03-10"]})
    doc = Doctor(df)
    result = doc.convert_types()
    assert pd.api.types.is_datetime64_any_dtype(result["joined"])


def test_detect_outliers_iqr():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    doc = Doctor(df)
    outliers = doc.detect_outliers(method="iqr")
    assert 5 in outliers["x"]


def test_treat_outliers_winsorizes():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    doc = Doctor(df)
    result = doc.treat_outliers(method="iqr")
    assert result["x"].max() < 100


def test_treat_aggressive_caps_outliers():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    doc = Doctor(df)
    cleaned = doc.treat(strategy="aggressive")
    assert cleaned["x"].max() < 100


def test_invalid_input_raises():
    with pytest.raises(TypeError):
        Doctor([1, 2, 3])