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

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Now use resource_path when you need to access bundled files:
PROMPT_TEMPLATE = resource_path('garmin_activity_tracker/prompt_template.txt')

def AI_format(df, df_splits,df_tss, config):

    total_summary_files = int(config.get("AI_format", {}).get("total_summary_files",50))
    total_split_files = int(config.get("AI_format", {}).get("total_split_files",3))

    dfInput = df.head(total_summary_files).copy().reset_index()
    dataSimple = pd.DataFrame({})
    dataSimple['distance [miles]'] = dfInput['distance']
    dataSimple['duration [minutes]'] = np.round(dfInput['elapsedDuration']/60,1)
    dataSimple['avg pace [min per mile]'] = np.round(dataSimple['duration [minutes]'] / dataSimple['distance [miles]'],1)
    dataSimple['avg Heart Rate'] = np.round(dfInput['averageHR'],1)
    date_obj = dfInput['startTimeLocal']
    dataSimple['Date'] = dfInput['startTimeLocal'].dt.strftime('%Y-%m-%d')
    df_tss_copy = df_tss.copy()
    df_tss_copy['Date'] = df_tss_copy['startTimeLocal'].dt.strftime('%Y-%m-%d')

    tss_cols = ['Date', 'TSS']  # adjust as needed
    dataSimple = dataSimple.merge(df_tss_copy [tss_cols], on='Date', how='left')
    dataSimple['Workout Type'] = dfInput['trainingEffectLabel']
    csv_data = dataSimple.round(1).to_csv(index=False)
    currentTime = datetime.datetime.now().isoformat()

    numSplits = total_split_files
    splitSimple = pd.DataFrame({})  
    latest_ids = df.iloc[0:numSplits]['activityId']
    dfTemp = pd.DataFrame({})
    for latest_id in latest_ids:
        print(f"Latest activity ID: {latest_id}")
        latest_splits = df_splits[df_splits['activityId'].astype(str) == str(latest_id)]
        date_obj = df[df['activityId'] == latest_id].index[0].date()
        dfTemp = latest_splits[['duration','distance']]*np.array((1/60, 0.00062137))
        dfTemp['Date'] = date_obj.strftime('%Y-%m-%d')
        dfTemp['Pace [min per mile]'] = dfTemp['duration'] / dfTemp['distance']
        #dfTemp['Pace [mi/hr]'] = dfTemp['distance'] / dfTemp['duration']*60
        #dfTemp['Elevation Gain [ft]'] = latest_splits['elevationGain']
        #dfTemp['Elevation Loss [ft]'] = latest_splits['elevationLoss']
        dfTemp['Elevation Gain [ft]'] = (latest_splits['elevationGain'] - latest_splits['elevationLoss'])*3.28084  # Convert meters to feet
        dfTemp['Average HR'] = latest_splits['averageHR']
        splitSimple = pd.concat([splitSimple, dfTemp], ignore_index=True)
    splitSimple['duration'] = splitSimple['duration']*60
    splitSimple['distance'] = splitSimple['distance'].round(2)
    splitSimple = splitSimple.rename(columns={'duration':'duration [seconds]'})
    splitSimple = splitSimple.rename(columns={'distance':'distance [miles]'})
    # Sort by split number if needed (assumes 'lapIndex' or similar exists)

    groupedSplits = splitSimple.groupby('Date')
    splitString = []
    for date in sorted(groupedSplits.groups.keys(), reverse=True):
        group = groupedSplits.get_group(date)
        splitString.append(f"Activity Date: {date}")
        # Format the splits for this date as a table, without the Date column (since it's redundant here)
        group_display = group.drop(columns=['Date'], errors='ignore').round(1).to_string(index=False)
        splitString.append(group_display)
        splitString.append("")  # Add a blank line between dates

    # Join all split summaries into a single string
    splits_summary = "\n".join(splitString)
    historical_summary = get_historical_summary(df,config)
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
        return error_msg, csv_data  # Or (None, None) if you want

    print("Prompt sent to Ollama:\n", prompt_content)
    return(prompt_content, csv_data)

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
    # Ensure datetime

    hms_threshold_limit = int(config.get("AI_format", {}).get("hms_threshold_limit", 90)) # in minutes
    df = df.copy()
    df['year'] = df.index.year
    df['week'] = df.index.isocalendar().week  # <-- Add here, before slicing

    all_years = sorted(df['year'].unique())
    years_to_show = all_years[-max_years:]  # last N years

    summary_lines = []
    for year in years_to_show:
        df_year = df[df['year'] == year]

        # Total miles
        total_miles = df_year['distance'].sum().round()

        # Find sub-90 and sub-88 half marathons
        # Assume HM is identified by 'activityName' or similar, and time is in 'elapsedDuration' (seconds)
        # 13.1 miles = HM
        marathons = df_year[(df_year['Race']=='race') & (df_year['distance'] >= 26.0) & (df_year['distance'] <= 26.5)]
        hms = df_year[(df_year['Race']=='race') & (df_year['distance'] >= 13) & (df_year['distance'] <= 13.3)]
        #hms = df_year[(df_year['distance'] >= 13) & (df_year['distance'] <= 13.3)]
        hms_thresh = hms[hms['elapsedDuration'] <= hms_threshold_limit*60]

        
        # Average TSS per week
        if 'TSS' in df_year.columns:
            if 'week' not in df_year.columns:
                df_year['week'] = df_year.index.isocalendar().week
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
        else:
            parts.append(f"{len(hms)} half marathons with average time of {hms['duration'].mean() // 60} minutes")
        if len(marathons) > 0:
            parts.append(f"{len(marathons)}x marathon with average time of {marathons['duration'].mean() // 60} minutes")
        if avg_tss_week > 0:
            parts.append(f"avg TSS/week ≈ {avg_tss_week}")
        if peak_week > 0:
            parts.append(f"peak week: {peak_week}")

        line = ", ".join(parts)
        summary_lines.append(line)
    return "Historical Summary\n" + "\n".join(summary_lines)

