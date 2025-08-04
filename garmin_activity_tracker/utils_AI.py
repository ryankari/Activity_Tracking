"""
Filename: utils_AI.py
Description:  Utility functions for AI interactions in Garmin Activity Tracker - formatting data and generating prompts.
Author: Ryan Kari
License: MIT
Created: 2025-07-20
"""

from ollama import chat
from ollama import ChatResponse
import os
import pandas as pd 
import numpy as np
import datetime
from jinja2 import Template
import sys
import subprocess

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Now use resource_path when you need to access bundled files:
PROMPT_TEMPLATE = resource_path('garmin_activity_tracker/prompt_template.txt')


def get_ollama_models(self):
    for cmd in ["ollama", "ollama.exe"]:
        try:
            result = subprocess.run(
                [cmd, "list"], capture_output=True, text=True, check=True
            )
            models = []
            for line in result.stdout.splitlines()[1:]:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        except Exception as e:
            continue
    print("Error listing Ollama models: ollama not found in PATH")
    return []


def AI_format(activities, config):
    """
    Format activity data for AI analysis using the new activities dictionary structure.
    
    Parameters:
        activities: Dictionary containing activity data (e.g., activities['Running']['Summary'])
        config: Configuration dictionary
    
    Returns:
        tuple: (prompt_content, csv_data)
    """
    total_summary_files = int(config.get("AI_format", {}).get("total_summary_files", 50))
    total_split_files = int(config.get("AI_format", {}).get("total_split_files", 3))

    # Extract running data from activities dictionary
    df = activities.get('Running', {}).get('Summary')
    df_splits = activities.get('Running', {}).get('Splits')
    df_tss = activities.get('Running', {}).get('TSS')
    
    if df is None or df.empty:
        error_msg = "ERROR: No running activity data available for AI formatting."
        return error_msg, None

    dfInput = df.head(total_summary_files).copy().reset_index()
    dataSimple = pd.DataFrame({})
    dataSimple['distance [miles]'] = dfInput['distance']
    dataSimple['duration [minutes]'] = np.round(dfInput['elapsedDuration']/60, 1)
    dataSimple['avg pace [min per mile]'] = np.round(dataSimple['duration [minutes]'] / dataSimple['distance [miles]'], 1)
    dataSimple['avg Heart Rate'] = np.round(dfInput['averageHR'], 1)
    dataSimple['Date'] = dfInput['startTimeLocal'].dt.strftime('%Y-%m-%d')
    
    # Handle TSS data if available
    if df_tss is not None and not df_tss.empty:
        df_tss_copy = df_tss.copy()
        df_tss_copy['Date'] = df_tss_copy['startTimeLocal'].dt.strftime('%Y-%m-%d')
        tss_cols = ['Date', 'TSS']
        dataSimple = dataSimple.merge(df_tss_copy[tss_cols], on='Date', how='left')
    else:
        # Add empty TSS column if no TSS data available
        dataSimple['TSS'] = np.nan
    dataSimple['Workout Type'] = dfInput['trainingEffectLabel']
    csv_data = dataSimple.round(1).to_csv(index=False)
    currentTime = datetime.datetime.now().isoformat()

    # Handle splits data if available
    splits_summary = ""
    if df_splits is not None and not df_splits.empty:
        numSplits = total_split_files
        splitSimple = pd.DataFrame({})  
        latest_ids = df.iloc[0:numSplits]['activityId']
        dfTemp = pd.DataFrame({})
        for latest_id in latest_ids:
            print(f"Latest activity ID: {latest_id}")
            latest_splits = df_splits[df_splits['activityId'].astype(str) == str(latest_id)]
            if not latest_splits.empty:
                date_obj = df[df['activityId'] == latest_id].index[0].date()
                dfTemp = latest_splits[['duration','distance']]*np.array((1/60, 0.00062137))
                dfTemp['Date'] = date_obj.strftime('%Y-%m-%d')
                dfTemp['Pace [min per mile]'] = dfTemp['duration'] / dfTemp['distance']
                dfTemp['Elevation Gain [ft]'] = (latest_splits['elevationGain'] - latest_splits['elevationLoss'])*3.28084  # Convert meters to feet
                dfTemp['Average HR'] = latest_splits['averageHR']
                splitSimple = pd.concat([splitSimple, dfTemp], ignore_index=True)
        
        if not splitSimple.empty:
            splitSimple['duration'] = splitSimple['duration']*60
            splitSimple['distance'] = splitSimple['distance'].round(2)
            splitSimple = splitSimple.rename(columns={'duration':'duration [seconds]'})
            splitSimple = splitSimple.rename(columns={'distance':'distance [miles]'})

            groupedSplits = splitSimple.groupby('Date')
            splitString = []
            for date in sorted(groupedSplits.groups.keys(), reverse=True):
                group = groupedSplits.get_group(date)
                splitString.append(f"Activity Date: {date}")
                group_display = group.drop(columns=['Date'], errors='ignore').round(1).to_string(index=False)
                splitString.append(group_display)
                splitString.append("")  # Add a blank line between dates

            splits_summary = "\n".join(splitString)
    
    if not splits_summary:
        splits_summary = "No splits data available for recent activities."
    historical_summary = get_historical_summary(df, config)
    prompt_content = load_prompt_template(
        PROMPT_TEMPLATE,
        currentTime=currentTime,
        csv_data=csv_data,
        splits_summary=splits_summary,
        total_split_files=total_split_files,
        historical_summary=historical_summary
    )

    if prompt_content is None:
        # Instead of print, return a special error message
        error_msg = (
            "ERROR: Prompt template file not found at: {}\n"
            "Please ensure 'prompt_template.txt' is present in the correct location."
        ).format(PROMPT_TEMPLATE)
        return error_msg, csv_data

    print("Prompt sent to Ollama:\n", prompt_content)
    return (prompt_content, csv_data)

def load_prompt_template(template_path, **kwargs):
    if not os.path.exists(template_path):
        return None  # Or return a string with your error message

    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = Template(f.read())
    return template.render(**kwargs)

def get_response(prompt,model='tinyllama'):
    response = chat(model=model,messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    print(response['message']['content'])
    return response['message']['content']
    

def get_historical_summary(df, config, max_years=10):
    """
    Generate a historical summary of running activities.
    
    Parameters:
        df: DataFrame containing running activity data
        config: Configuration dictionary
        max_years: Maximum number of years to include in summary
    
    Returns:
        str: Formatted historical summary
    """
    if df is None or df.empty:
        return "Historical Summary\nNo historical data available."
        
    hms_threshold_limit = int(config.get("AI_format", {}).get("hms_threshold_limit", 90)) # in minutes
    df = df.copy()
    df['year'] = df.index.year
    df['week'] = df.index.isocalendar().week

    all_years = sorted(df['year'].unique())
    years_to_show = all_years[-max_years:]  # last N years

    summary_lines = []
    for year in years_to_show:
        df_year = df[df['year'] == year]

        # Total miles
        total_miles = df_year['distance'].sum().round()

        # Find marathons and half marathons
        marathons = df_year[(df_year['Race']=='race') & (df_year['distance'] >= 26.0) & (df_year['distance'] <= 26.5)]
        hms = df_year[(df_year['Race']=='race') & (df_year['distance'] >= 13) & (df_year['distance'] <= 13.3)]
        hms_thresh = hms[hms['elapsedDuration'] <= hms_threshold_limit*60]

        # Average TSS per week
        if 'TSS' in df_year.columns:
            tss_per_week = df_year.groupby('week')['TSS'].sum()
            avg_tss_week = int(np.round(tss_per_week.mean()))
            peak_week = int(np.round(tss_per_week.max()))
        else:
            avg_tss_week = peak_week = 0

        # Build line dynamically
        parts = [
            f"{year}: ~{int(total_miles)} miles",
        ]
        if len(hms_thresh) > 0:
            parts.append(f"{len(hms_thresh)}x sub-{hms_threshold_limit} min HM")
        elif len(hms) > 0:
            avg_hm_time = hms['elapsedDuration'].mean() / 60  # Convert to minutes
            parts.append(f"{len(hms)} half marathons with average time of {int(avg_hm_time)} minutes")
        
        if len(marathons) > 0:
            avg_marathon_time = marathons['elapsedDuration'].mean() / 60  # Convert to minutes
            parts.append(f"{len(marathons)}x marathon with average time of {int(avg_marathon_time)} minutes")
        
        if avg_tss_week > 0:
            parts.append(f"avg TSS/week ≈ {avg_tss_week}")
        if peak_week > 0:
            parts.append(f"peak week: {peak_week}")

        line = ", ".join(parts)
        summary_lines.append(line)
    
    return "Historical Summary\n" + "\n".join(summary_lines)

