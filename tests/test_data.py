"""Unit tests for data loading and cleaning."""
import numpy as np
import pandas as pd
import pytest

from heart_mlops import config, data


def test_load_raw_treats_question_marks_as_missing(tmp_path, raw_frame):
    path = tmp_path / "raw.csv"
    frame = raw_frame.astype(object)
    frame.loc[0, "ca"] = "?"
    frame.to_csv(path, index=False)
    loaded = data.load_raw(path)
    assert list(loaded.columns) == config.COLUMNS
    assert pd.isna(loaded.loc[0, "ca"])


def test_load_raw_rejects_missing_columns(tmp_path, raw_frame):
    path = tmp_path / "bad.csv"
    raw_frame.drop(columns=["chol"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="chol"):
        data.load_raw(path)


def test_binarise_target_maps_any_severity_to_one():
    df = pd.DataFrame({"num": [0, 1, 2, 3, 4]})
    out = data.binarise_target(df)
    assert out[config.TARGET].tolist() == [0, 1, 1, 1, 1]
    assert "num" not in out.columns


def test_flag_out_of_range_sets_nan(raw_frame):
    out = data.flag_out_of_range(raw_frame)
    assert np.isnan(out.loc[2, "trestbps"])
    assert out["age"].notna().all()


def test_missing_value_report_lists_only_gaps(raw_frame):
    rep = data.missing_value_report(raw_frame)
    assert set(rep.index) == {"ca", "thal"}
    assert (rep["missing"] == 1).all()


def test_clean_produces_complete_binary_dataset(raw_frame):
    out = data.clean(raw_frame)
    assert not out.isna().any().any()
    assert len(out) == len(raw_frame) - 1               # duplicate removed
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert out["trestbps"].between(70, 220).all()
    assert out["ca"].dtype.kind == "i"


def test_clean_is_idempotent_on_row_count(raw_frame):
    once = data.clean(raw_frame)
    assert len(once) == len(once.drop_duplicates())
