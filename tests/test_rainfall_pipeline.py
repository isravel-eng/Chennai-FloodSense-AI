import os
import pytest
import pandas as pd
from model_1_rainfall.locality_forecast import forecast_locality, load_locality_monthly
from pipeline.backfill_openmeteo import MASTER_CSV

def test_pipeline_idempotency_and_no_duplicates():
    df = pd.read_csv(MASTER_CSV)
    # Check for duplicates across date and locality
    duplicates = df[df.duplicated(subset=["date", "locality"])]
    assert duplicates.empty, f"Found {len(duplicates)} duplicate rows in master_dataset.csv!"

def test_complete_month_exclusion():
    # Sholinganallur was backfilled. Its last date is yesterday.
    # So its training_end should be the month prior to this month.
    res = forecast_locality("Zone 14 U41 Perungudi", 12)
    
    assert res["status"] in ["ok", "fallback_forecast"]
    
    training_end = pd.Timestamp(res["training_end"])
    forecast_start = pd.Timestamp(res["forecast_start"])
    
    # forecast_start should be strictly exactly 1 month after training_end
    assert forecast_start == training_end + pd.DateOffset(months=1)
    
    # The training_end month should be strictly less than the current calendar month
    current_month_start = pd.Timestamp.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    assert training_end < current_month_start, "training_end should not include the current partial month"
    
def test_forecast_dates_not_relabeled():
    res = forecast_locality("Zone 14 U41 Perungudi", 12)
    forecast_start = pd.Timestamp(res["forecast_start"])
    first_forecast_month = pd.Timestamp(res["forecast"][0]["month"])
    
    assert first_forecast_month == forecast_start, "The first forecast month must equal the forecast_start"


def test_recent_contiguous_history_is_used_for_sparse_locality():
    result = forecast_locality("Alandur", 6)
    assert result["status"] == "ok"
    assert result["horizon_months"] == 6
    assert result["observed_months"] >= 24
    assert result["training_start"] >= "2024-01"
    assert result["training_end"] < pd.Timestamp.today().strftime("%Y-%m")
    assert len(result["forecast"]) == 6
