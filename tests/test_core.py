import os
import pytest
from garmin_activity_tracker.core import ActivityTracker

@pytest.fixture
def tracker():
    username = os.getenv("GARMIN_USERNAME")
    password = os.getenv("GARMIN_PASSWORD")
    if not username or not password:
        pytest.skip("GARMIN_USERNAME and GARMIN_PASSWORD must be set in environment")
    return ActivityTracker(username, password)

def test_sync_summary_data(tracker):
    df_summary = tracker.sync_summary_data()
    assert not df_summary.empty
    assert "activityId" in df_summary.columns

def test_sync_split_data(tracker):
    df_summary = tracker.sync_summary_data()
    df_splits = tracker.sync_split_data(df_summary)
    assert not df_splits.empty
    assert "activityId" in df_splits.columns

def test_preprocess_running_data(tracker):
    df_summary = tracker.sync_summary_data()
    df_running = tracker.preprocess_running_data(df_summary)
    assert not df_running.empty
    assert "Type" in df_running.columns
    assert all(df_running["Type"].str.lower() == "running") 

def test_calculate_tss(tracker):
    df_summary = tracker.sync_summary_data()
    df_running = tracker.preprocess_running_data(df_summary)
    df_tss = tracker.calculate_tss(df_running)
    assert not df_tss.empty
    assert "TSS" in df_tss.columns

def test_split_data_columns(tracker):
    df_summary = tracker.sync_summary_data()
    df_splits = tracker.sync_split_data(df_summary)
    expected_cols = {"activityId", "duration", "distance"}
    assert expected_cols.issubset(df_splits.columns)
    