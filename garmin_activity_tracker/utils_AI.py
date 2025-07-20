from ollama import chat
from ollama import ChatResponse
import os
import pandas as pd 
import numpy as np
import datetime

def AI_format(df, df_splits,df_tss,n=50):
    dfInput = df.head(n).copy().reset_index()
    dataSimple = pd.DataFrame({})
    #dataSimple['activityId'] = dfInput['activityId']
    #dataSimple['Date'] = dfInput.index
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

    csv_data = dataSimple.round(1).to_csv(index=False)
    currentTime = datetime.datetime.now().isoformat()

    numSplits = 3
    splitSimple = pd.DataFrame({})  
    latest_ids = df.iloc[0:numSplits]['activityId']
    dfTemp = pd.DataFrame({})
    for latest_id in latest_ids:
        print(f"Latest activity ID: {latest_id}")
        latest_splits = df_splits[df_splits['activityId'] == latest_id]
        date_obj = df[df['activityId'] == latest_id].index[0].date()
        dfTemp = latest_splits[['duration','distance']]*np.array((1/60, 0.00062137))
        dfTemp['Date'] = date_obj.strftime('%Y-%m-%d')
        dfTemp['Pace [min per mile]'] = dfTemp['duration'] / dfTemp['distance']
        dfTemp['Pace [mi/hr]'] = dfTemp['distance'] / dfTemp['duration']*60
        dfTemp['Average HR'] = latest_splits['averageHR']
        splitSimple = pd.concat([splitSimple, dfTemp], ignore_index=True)
    splitSimple = splitSimple.rename(columns={'duration':'duration [min]'})
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
                           
    prompt_content = (
        "Act as a half marathon coach for a local runner in the top 2%. I tend to adapt well to the Hal Higdon training plans."
        "My easy pace is around 8:30 per mile, and I can run a 5K in 19 minutes."
        "I adapt well to 20 percent hard efforts with ideally 80 percent in the easy ranges.\n"
        "I have completed 3 sub 90 minute half marathons per year the last 4 years.\n "
        "Given my recent running data (in CSV below), provide feedback on my average pace, "
        "training load, and whether my workouts suggest I am overtraining or on track for improvement.\n"
        "The current time is: "f"{currentTime}\n"
        f"{csv_data}\n"
        "The splits for the last 4 activities are as follows:\n"
        f"{splits_summary}\n"
        "In particular, comment on the last 2 runs and their splits.\n"
        "Consider this when suggesting what the next workout should be or how much rest I should get.\n" 
        "Please be specific and use the data provided. Try to keep the response under 1000 characters.\n"
    )

    prompt_content = (
        "You are a high-performance half marathon coach assisting a competitive age-group runner "
        "who consistently finishes in the top 2% and follows a Hal Higdon-style 80/20 training plan.\n"
        "Athlete profile:\n"
        "- Easy pace: ~8:30 min/mile\n"
        "- 5K best: 19:00\n"
        "- Half marathon PR: sub-90 minutes (x3/year)\n"
        "- Tolerates 20% hard effort, adapts well to progressive loading\n\n"

        "Your task is to analyze the runner's latest training data below and give:\n"
        "1. An evaluation of the training load (based on TSS and pacing)\n"
        "2. Feedback on average and recent pacing trends\n"
        "3. Insights into overtraining, fitness progression, or undertraining\n"
        "4. Specific feedback on the last 2 workouts and their splits\n"
        "5. A recommended next workout OR recovery period\n\n"

        f"Current time: {currentTime}\n\n"
        "Training summary (most recent 50 runs):\n"
        f"{csv_data}\n\n"
        "Recent split data (last 3 runs):\n"
        f"{splits_summary}\n\n"

        "Please be concise (under 1000 characters), specific, and data-informed."
    )

    print("Prompt sent to Ollama:\n", prompt_content)
    return(prompt_content, csv_data)



def get_response(prompt,model='tinyllama'):
    #response = chat(model='mistral', messages=[
    response = chat(model=model,messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    print(response['message']['content'])
    return response['message']['content']
    

