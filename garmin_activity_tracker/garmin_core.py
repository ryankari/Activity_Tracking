"""
Filename: core.py
Description: Core logic for Garmin Activity Tracker - data sync, processing, and TSS calculations.
Author: Ryan Kari
License: MIT
Created: 2025-07-20
"""
import os
import sys
import time
import pandas as pd
from garminconnect import Garmin
import logging
import datetime
import numpy as np
import json
import ast



def get_base_path():
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running as a script
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
DATA_DIR = os.path.join(BASE_PATH, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists

SUMMARY_FILE = os.path.join(DATA_DIR, "garminSummaryData.xlsx")
SPLITS_FILE = os.path.join(DATA_DIR, "garminSplitData.xlsx")
OTHER_DATA_FILE = os.path.join(DATA_DIR, "OtherData.xlsx")

class ActivityTracker :
    def __init__(self, username, password):
        """
        Initialize the ActivityTracker object with Garmin credentials.

        Args:
            username (str): Garmin Connect username.
            password (str): Garmin Connect password.
        """
        self.username = username
        self.password = password

    def load_column_map(self,filename="signal_map.json"):
        path = os.path.join(BASE_PATH, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            return mapping["columns"]
        except Exception as e:
            logging.error(f"Error loading signal_map.json: {e}")
            return {}

    def sync_summary_data(self, max_activities=3000, batch_size=100, use_api=True):
        """
        Sync activity summary data from the Garmin API or load from local Excel.

        Args:
            max_activities (int): Maximum number of activities to download.
            batch_size (int): Number of activities per API call.
            use_api (bool): If False, only load from Excel.

        Returns:
            pd.DataFrame: Activity summary data.
        """
        # Use mapped column names
        column_map = self.load_column_map()  # Load mapping
        col_start_time = column_map["startTimeLocal"]
        other_start_time = column_map["otherStartTime"]
        col_activity_id = column_map["activityId"]

        if os.path.exists(SUMMARY_FILE):
            df_existing = pd.read_excel(SUMMARY_FILE)
            if col_start_time in df_existing.columns:
                df_existing[col_start_time] = pd.to_datetime(df_existing[col_start_time], errors='coerce')

            existing_ids = set(
                df_existing[col_activity_id]
                .dropna()                      # remove NaN
                .astype(float)                # ensure float type first (if Excel added .0)
                .astype(int)                  # safely cast to int
                .astype(str)                  # convert to string
)
        else:
            df_existing = pd.DataFrame()
            existing_ids = set()
            use_api = True



        if not use_api:
            return df_existing

        client = Garmin(self.username, self.password)
        client.login()


        new_activities = []
        newActivitiesExist = False
        for start in range(0, max_activities, batch_size):
            activities = client.get_activities(start, batch_size)
            if not activities:
                break
            for activity in activities:
                aid = str(activity.get(col_activity_id))
                if aid not in existing_ids:
                    print("Appending new activity with ID: {}".format(aid))
                    newActivitiesExist = True
                    new_activities.append(activity)


        df_combined = pd.concat([pd.DataFrame(new_activities), df_existing], ignore_index=True)

        if newActivitiesExist:
            df_combined.to_excel(SUMMARY_FILE, index=False)

        if 'df_combined' in locals():
            outputVar = df_combined
        else:
            outputVar = df_existing

        if os.path.exists(OTHER_DATA_FILE):
            df_other = pd.read_excel(OTHER_DATA_FILE)
            if other_start_time in df_other.columns:
                df_other[other_start_time] = pd.to_datetime(df_other[other_start_time], errors='coerce')
            print(f"Loaded {len(df_other)} old activities from {OTHER_DATA_FILE}.")
            df_combined = pd.concat([outputVar, df_other], ignore_index=True)


        print("Data synced with df_summary with length = {} records".format(len(outputVar)))
        return outputVar

    def sync_split_data(self, df_summary, n=30000, use_api=True):
        """
        Sync recent split data (laps) from Garmin or load from local file.

        Args:
            df_summary (pd.DataFrame): DataFrame with summary activities.
            n (int): Number of most recent runs to fetch splits for.
            use_api (bool): If False, only load from Excel.

        Returns:
            pd.DataFrame: Combined split data.
        """
        column_map = self.load_column_map()  # Load mapping
        col_start_time = column_map["startTimeLocal"]
        other_start_time = column_map["otherStartTime"]
        col_activity_id = column_map["activityId"]

        if os.path.exists(SPLITS_FILE):
            df_existing = pd.read_excel(SPLITS_FILE)
            if col_start_time in df_existing.columns:
                df_existing[col_start_time] = pd.to_datetime(df_existing[col_start_time], errors='coerce')
            existing_ids = set(df_existing[col_activity_id].astype(str))
        else:
            df_existing = pd.DataFrame()
            existing_ids = set()

        if not use_api:
            return df_existing


        client = Garmin(self.username, self.password)
        client.login()

        new_splits = []
        df_summary = df_summary.reset_index()
        df_summary[col_start_time] = pd.to_datetime(df_summary[col_start_time], errors='coerce')
        df_sorted = df_summary.sort_values(by=col_start_time, ascending=False).head(n)
        if df_existing.empty:
            print("No summary data available to fetch splits.")
            for _, row in df_sorted.iterrows():
                
                try:
                    aid = str(int(row.get(col_activity_id )))
                    print("Requested activityId {}".format(aid))
                    details = client.get_activity_splits(aid)
                    print("New splits found for activityId {}".format(aid))
                    for lap in details.get("lapDTOs", []):
                        lap[col_activity_id ] = aid
                        new_splits.append(lap)
                except Exception as e:
                    print(f"No existing file. Failed to get splits for {aid}: {e}")
                    pass
            df_combined = pd.concat([pd.DataFrame(new_splits), df_existing], ignore_index=True)
            df_combined.to_excel(SPLITS_FILE, index=False)

            return df_existing
        else:
            
            for _, row in df_sorted.iterrows():

                try:
                    rowValue = row.get(col_activity_id )
                    if pd.isna(rowValue):
                        print("Skipping NaN activityId")
                        
                    else:
                        aid = str(int(rowValue))
                        if aid not in existing_ids:
                            print("Requested activityId {}".format(aid))
                            details = client.get_activity_splits(int(aid))
                            for lap in details.get("lapDTOs", []):
                                lap[col_activity_id] = aid
                                new_splits.append(lap)
                except Exception as e:
                    print(f"Existing data file. Failed to get splits for {rowValue}: {e}")
                    pass

            df_combined = pd.concat([pd.DataFrame(new_splits), df_existing], ignore_index=True)
            
            if not new_splits:
                print("No new splits found for the latest activities.")
            else:
                print(f"Found {len(new_splits)} new splits for the latest activities.")
                df_combined.to_excel(SPLITS_FILE, index=False)
            return df_combined

    def extract_typekey_list(self,series):
        import ast
        result = []
        for index, entry in enumerate(series):
            if isinstance(entry, str):
                try:
                    parsed = ast.literal_eval(entry)
                    result.append(parsed.get('typeKey', None))
                except Exception as e:
                    print(f"Literal eval error at {index}: {e}")
                    result.append(None)
            elif isinstance(entry, dict):
                result.append(entry.get('typeKey', None))
            else:
                print(f"Unknown type at {index}: {entry}")
                result.append(None)
        return result

#***
    def estimate_rTSS_miles(self,distance_mi, elevation_gain_ft, elevation_loss_ft,
                            avg_pace_min_per_mi, threshold_pace_min_per_mi = 6.25,
                            gain_factor=0.0005, loss_factor=0.00015):
        equiv_dist_mi = distance_mi + (elevation_gain_ft * gain_factor) + (elevation_loss_ft * loss_factor)
        duration_hr = equiv_dist_mi * avg_pace_min_per_mi / 60
        IF = threshold_pace_min_per_mi / avg_pace_min_per_mi
        rTSS = duration_hr * (IF ** 2) * 100
        return rTSS


    def preprocess_running_data(self, df):
        
        activity_list = df['activityType'].tolist()
        
        df['Type'] = self.extract_typekey_list(df['activityType'])
        df['Race'] = self.extract_typekey_list(df['eventType'])
        
        
        df = df[(df['Type'].str.lower() == 'running')].reset_index(drop=True)
        Pace = []
        VO2 = []
        TSSArray = []
        df['duration_str'] = df['duration'].apply(lambda x: str(datetime.timedelta(seconds=int(x))))
        
        df['startTimeLocal'] = pd.to_datetime(df['startTimeLocal'], errors='coerce')
        df = df.set_index('startTimeLocal')
        df['distance'] = df['distance'] * 0.00062137

        for index, row in df.iterrows():
            try:
                if isinstance(row['duration'], str):
                    t = datetime.datetime.strptime(row['duration'], '%H:%M:%S')


                total_min = row['duration'] / 60
                pace = total_min / row['distance']
                vo2 = 108.844 - 0.1636 * (170 / 2.2) - (1.438 * pace) - (0.1928 * float(row['averageHR']))
                TSS = self.estimate_rTSS_miles(
                    distance_mi=row['distance'],
                    elevation_gain_ft=row.get('elevationGain', 0),
                    elevation_loss_ft=row.get('elevationLoss', 0),
                    avg_pace_min_per_mi=pace
                )
                Pace.append(pace)
                VO2.append(vo2)
                TSSArray.append(TSS)
            except Exception as e:
                print(f"Error on row {index}: {e}")
                Pace.append(np.nan)
                VO2.append(np.nan)

        df['Avg Pace'] = Pace
        df['Avg HR'] = df['averageHR'].astype(float)
        df['Avg Cadence'] = df['maxDoubleCadence'].astype(float)
        df['VO2'] = VO2
        df['Velocity (m/s)'] = 1609.34 / (df['Avg Pace'] * 60)
        df['TSS'] = TSSArray    
        return df

    def calculate_tss(self, df):
        df_tss = df.reset_index()
        df_tss = df_tss[['startTimeLocal', 'TSS']].copy()  # Extract TSS data
        df_tss['startTimeLocal'] = pd.to_datetime(df_tss['startTimeLocal']).dt.floor('D')

        full_dates = pd.DataFrame({'startTimeLocal': pd.date_range(df_tss['startTimeLocal'].min(), df_tss['startTimeLocal'].max())})
        df_daily = pd.merge(full_dates, df_tss, on='startTimeLocal', how='left')
        df_daily['TSS'] = df_daily['TSS'].fillna(0.0)

        CTL_CONST = 42
        ATL_CONST = 7

        ctl = atl = df_daily.loc[0, 'TSS']
        df_daily['ctl'] = 0.0
        df_daily['atl'] = 0.0
        df_daily['tsb'] = 0.0

        for i in range(len(df_daily)):
            tss_today = df_daily.loc[i, 'TSS']
            ctl += (tss_today - ctl) * (1 / CTL_CONST)
            atl += (tss_today - atl) * (1 / ATL_CONST)
            tsb = ctl - atl
            df_daily.loc[i, 'ctl'] = ctl
            df_daily.loc[i, 'atl'] = atl
            df_daily.loc[i, 'tsb'] = tsb
        return df_daily


    def _parse_activity_and_race_labels(self, df):
        """
        Extract 'Type' and 'Race' from nested Garmin dictionaries in activityType and eventType.

        Args:
            df (pd.DataFrame): Input DataFrame with 'activityType' and 'eventType'.

        Returns:
            pd.DataFrame: DataFrame with 'Type' and 'Race' columns added.
        """
        def parse_entry(entry):
            if isinstance(entry, str):
                try:
                    return ast.literal_eval(entry).get("typeKey", None)
                except:
                    return None
            elif isinstance(entry, dict):
                return entry.get("typeKey", None)
            return None

        df['Type'] = df['activityType'].apply(parse_entry)
        df['Race'] = df['eventType'].apply(parse_entry)
        return df
