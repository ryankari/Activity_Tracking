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