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

def AI_format(df, df_splits,df_tss,n=50):
    dfInput = df.head(n).copy().reset_index()
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

    numSplits = 3
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

    prompt_content = load_prompt_template(
    PROMPT_TEMPLATE,
    currentTime=currentTime,
    csv_data=csv_data,
    splits_summary=splits_summary
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
    

