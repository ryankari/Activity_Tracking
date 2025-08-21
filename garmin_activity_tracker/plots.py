"""
Filename: plots.py
Description: Plotting functions for Garmin Activity Tracker - visualizations of activity data.
Author: Ryan Kari
License: MIT
Created: 2025-07-20
"""
from cProfile import label
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import calendar


def create_basic_metric_plot(df, ax, show_trend=True):
    

    distance_per_month = df.resample('ME')['distance'].sum()

    plot_items = [
        {'y': df['distance'], 'label': 'Distance [mi]', 'x': df.index, 'windowSize': 200},
        {'y': df['averageHR'], 'label': 'HR', 'x': df.index, 'windowSize': 200},
        {'y': distance_per_month, 'label': 'Distance [mi monthly]', 'x': distance_per_month.index, 'windowSize': 30}
    ]

    for i, item in enumerate(plot_items):
        ax[i].plot(item['x'], item['y'], '.', markersize=8)
        if show_trend:
            rolling_mean = item['y'].rolling(window=item['windowSize'], center=False, min_periods=20).mean()
            ax[i].plot(
                item['x'][rolling_mean.notna()],
                rolling_mean.dropna(),
                color='black', linewidth=2, label='Trend'
            )
        ax[i].set_ylabel(item['label'])
        ax[i].grid(True)

    ax[-1].xaxis.set_major_locator(mdates.YearLocator(1))
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax[-1].xaxis.get_majorticklabels(), rotation=45)

    latest_date = df.index.max()
    earliest_date = latest_date - pd.DateOffset(years=10)
    # Set x-axis limits for all subplots
    for axis in ax:
        axis.set_xlim([earliest_date, latest_date])

    plt.tight_layout()
    #plt.show()

def create_summary_plot(df_summary):
    if 'startTimeLocal' in df_summary.columns:
        df_summary['startTimeLocal'] = pd.to_datetime(df_summary['startTimeLocal'], errors='coerce')
        df_summary = df_summary.set_index('startTimeLocal')
    if 'distance' not in df_summary.columns:
        print("Missing 'distance' column.")
        return
    monthly_distance = df_summary.resample('ME')['distance'].sum()
    create_basic_metric_plot(df_summary, monthly_distance)




def createAdvancedMetricPlot(df, ax, show_trend=True):
    distance_per_month = df.resample('ME')['distance'].sum()
    monthlyFastest = df.resample('ME')['Avg Pace'].agg(['min'])


    latest_date = df.index.max()
    earliest_date = latest_date - pd.DateOffset(years=10)
    # Set x-axis limits for all subplots
    for axis in ax:
        axis.set_xlim([earliest_date, latest_date])

    ax[1].plot(monthlyFastest, '.')
    ax[1].set_ylabel('Fastest avg pace [min/mi]')
    ax[1].grid(True)

    
    ax[0].plot(distance_per_month)
    ax[0].set_ylabel('Distance [mi per month]')
    ax[0].grid(True)
    ax[0].xaxis.set_major_locator(mdates.YearLocator(1))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax[2].plot(df.index, df['VO2'], '.', markersize=8)
    ax[2].plot(df.index, df['VO2'].rolling(window=200, min_periods=20).mean(), color='black', linewidth=2)
    ax[2].set_ylabel('VO2 [avg]')
    ax[2].grid(True)
    ax[2].xaxis.set_major_locator(mdates.YearLocator(1))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def plot_race_performance(df_summary,ax,config=None):
    if 'Race' not in df_summary.columns or 'Avg Pace' not in df_summary.columns:
        print("Race or Avg Pace data not available.")
        return
    races = df_summary[df_summary['Race'].str.lower() == 'race']
    #fig = plt.figure(figsize=(10, 8))
    ax.plot(races[races['distance'] < 26.2]['Avg Pace'], 'or', label='< Marathon')
    ax.plot(races[races['distance'] >= 26.2]['Avg Pace'], 'ob', label='≥ Marathon')
    ax.plot(races[races['distance'] < 10]['Avg Pace'], 'og', label='< 10mi')

    previous_date = races.index[0]
    for index, item in enumerate(races.iterrows()):
        delta = 0.1
        if ((previous_date - item[0]).total_seconds() < 864000) and index > 0:
            delta = 1.0
        ax.text(item[0], item[1]['Avg Pace'] + delta, item[1]['activityName'][:20], rotation=90, fontsize=8)
        previous_date = item[0]

    #ax.set_title('Race Outcomes')
    ax.grid(True)
    ax.set_ylabel('min/mi')
    ax.set_xlabel('Event Index')
    min = float(config.get("plotting", {}).get("race_ylim_min", 4.0))
    max = float(config.get("plotting", {}).get("race_ylim_max", 10.0))
    ax.set_ylim([min, max])
    #plt.legend()
    #ax.tight_layout()
    #plt.show()

def distance_to_size(distance, min_dist=2, max_dist=13, min_size=200, max_size=1000):
    """
    Linearly maps distance [min_dist, max_dist] to size [min_size, max_size].
    Anything above max_dist gets max_size.
    """
    distance = max(min_dist, min(distance, max_dist))
    distanceScaled = distance

    if distance > max_dist:
        distanceScaled = max_dist
    if distance < min_dist:
        distanceScaled = min_dist
    # Linear interpolation
    sizeOutput = min_size + (max_size - min_size) * ((distanceScaled - min_dist) / (max_dist - min_dist))
    return sizeOutput
def get_activity_style(activity_type, row, config):
    """
    Returns style information (color, size, label) for different activity types.
    """
    style_cfg = config.get("activity_styles", {}).get(activity_type, {})
    # Handle race color for running
    if activity_type == "Running" and str(row.get('Race', '')).lower() == 'race':
        color = style_cfg.get("color_race", "red")
    else:
        color = style_cfg.get("color", "skyblue")
    size_metric = style_cfg.get("size_metric")
    default_size = style_cfg.get("default_size", 300)
    label_metric = style_cfg.get("label_metric", "distance")
    label_format = style_cfg.get("label_format", "{:.1f}")

    # Calculate size
    if size_metric == "distance" and activity_type == "Running":
        size = distance_to_size(row['distance'])
    else:
        size = default_size

    # Get label value
    if label_metric in row:
        label_value = row[label_metric]
        if label_metric == "duration":
            label_value = label_value / 60  # Convert seconds to minutes
        label = label_format.format(label_value)
    else:
        label = ""

    return {
        "color": color,
        "size": size,
        "label": label,
        "metric": label_metric
    }

def add_activity_circle(ax, x, y, row, activity_type, config, training_effect_length):
    """
    Adds a circle representing an activity to the plot.
    
    Parameters:
        ax: matplotlib axis
        x, y: position coordinates
        row: DataFrame row containing activity data
        activity_type: String indicating activity type
        config: Configuration dictionary
        training_effect_length: Maximum length for training effect labels
    """
    style = get_activity_style(activity_type, row, config)
    
    # Plot the circle
    ax.scatter(x, y, s=style['size'], color=style['color'], 
              edgecolor='k', alpha=0.7)
    
    # Add metric label inside the circle
    if style['label']:
        ax.text(x, y, style['label'], ha='center', va='center', fontsize=8)
    
    # Add training effect label if available
    training_effect_label = row.get("trainingEffectLabel", "")
    if isinstance(training_effect_label, str) and training_effect_label:
        ax.text(x, y + 0.25, training_effect_label[:training_effect_length],
                ha='center', va='top', fontsize=8, color='black')


def plot_calendar(activities, ax, config, year=None, month=None):
    """
    Plots a calendar-style grid for a selected month (or last 4 weeks if no month/year given).
    Each activity is a circle sized by distance (for running only), color-coded by activity type.
    Handles months with more than 28 days.
    """
    # Extract activity dataframes based on config
    activity_data = {}
    activity_types = ['Running', 'Cycling', 'Swimming', 'Workouts']
    
    for activity_type in activity_types:
        config_key = f"include{activity_type}"
        checkTypebool = config.get("calendarplot", {}).get(config_key)
        if checkTypebool:
            df = activities.get(activity_type, {}).get('Summary')
            if df is not None and not df.empty:
                activity_data[activity_type] = df

    if not activity_data:
        print("No activities available for calendar plot.")
        return {}

    datePosition = float(config.get("calendarplot", {}).get("datePosition", 0.5))
    trainingEffectLength = int(config.get("calendarplot", {}).get("trainingEffectLength", 10))
    cell_to_date = {}  # (x, y) -> date
    if year is not None and month is not None:
        # Month calendar grid
        first_day = pd.Timestamp(year=year, month=month, day=1)
        last_day = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
        
        # Combine all activities for the month
        all_month_activities = []
        for activity_type, df in activity_data.items():
            df_month = df[(df.index >= first_day) & (df.index < last_day + pd.Timedelta(days=1))].copy()
            df_month['date'] = df_month.index.date
            df_month['activity_type'] = activity_type
            all_month_activities.append(df_month)
        
        if all_month_activities:
            df_all_month = pd.concat(all_month_activities, ignore_index=False)
        else:
            df_all_month = pd.DataFrame()

        # Use calendar.monthcalendar to get the grid (weeks x 7 days)
        month_calendar = calendar.monthcalendar(year, month)
        n_weeks = len(month_calendar)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, n_weeks - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(7))
        ax.set_yticks(range(n_weeks))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        # Calculate total distance for each week (running activities only)
        week_totals = []
        for i, week in enumerate(month_calendar):
            week_dates = [pd.Timestamp(year=year, month=month, day=day).date()
                            for day in week if day != 0]
            if 'Running' in activity_data and not df_all_month.empty:
                week_dist = df_all_month[
                    (df_all_month['date'].isin(week_dates)) & 
                    (df_all_month['activity_type'] == 'Running')
                ]['distance'].sum()
            else:
                week_dist = 0
            week_totals.append(week_dist)
        ax.set_yticklabels([f"Week {i+1}\n{week_totals[i]:.1f} mi" for i in range(n_weeks)])

        # Map each date to (week, day) and plot activities
        for i, week in enumerate(month_calendar):
            for j, day in enumerate(week):
                if day == 0:
                    continue  # No day in this cell
                day_date = pd.Timestamp(year=year, month=month, day=day).date()
                cell_to_date[(j, i)] = day_date
                
                # Add the date label in the cell
                ax.text(j, i+datePosition, pd.Timestamp(day_date).strftime('%b %d'),
                        ha='center', va='top', fontsize=10, color='black', fontweight='bold')
                
                # Get all activities for this day
                day_acts = df_all_month[df_all_month['date'] == day_date] if not df_all_month.empty else pd.DataFrame()
                n_acts = len(day_acts)
                if n_acts == 0:
                    continue
                
                # Calculate offsets for multiple activities
                if n_acts == 1:
                    offsets = np.linspace(j, j + 0.2, n_acts)
                else:
                    offsets = np.linspace(j - 0.2, j + 0.2, n_acts)
                
                # Plot each activity
                for k, (_, row) in enumerate(day_acts.iterrows()):
                    activity_type = row['activity_type']
                    add_activity_circle(ax, offsets[k], i, row, activity_type, config, trainingEffectLength)
        # Draw grid
        for x in range(8):
            ax.axvline(x-0.5, color='gray', lw=0.5)
        for y in range(n_weeks + 1):
            ax.axhline(y-0.5, color='gray', lw=0.5)

        ax.set_title(f'Activities: {calendar.month_name[month]} {year} (Calendar View)')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout()
        

    else:
        # Default: last 28 days ending with current date
        end_date = pd.Timestamp(pd.Timestamp.now().date())
        start_date = end_date - pd.Timedelta(days=27)
        
        # Combine all activities for the period
        all_period_activities = []
        for activity_type, df in activity_data.items():
            df_period = df[(df.index >= start_date) & (df.index < end_date + pd.Timedelta(days=1))].copy()
            df_period['date'] = df_period.index.date
            df_period['activity_type'] = activity_type
            all_period_activities.append(df_period)
        
        if all_period_activities:
            df_all_period = pd.concat(all_period_activities, ignore_index=False)
        else:
            df_all_period = pd.DataFrame()
        
        days = config.get("calendarplot", {}).get("calendar_days_shown", 28)
        all_dates = [end_date - pd.Timedelta(days=x) for x in reversed(range(days))]
        all_dates = [d.date() for d in all_dates]
        n_weeks = 4
        
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, n_weeks - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(7))
        ax.set_yticks(range(4))
        weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        first_weekday = pd.Timestamp(all_dates[0]).weekday()
        rotated_labels = weekday_labels[first_weekday:] + weekday_labels[:first_weekday]
        ax.set_xticklabels(rotated_labels)

        # Calculate weekly totals (running activities only)
        week_totals = []
        for week in range(4):
            week_start_idx = week * 7
            week_dates = all_dates[week_start_idx:week_start_idx + 7]
            if 'Running' in activity_data:
                week_dist = df_all_period[
                    (df_all_period['date'].isin(week_dates)) & 
                    (df_all_period['activity_type'] == 'Running')
                ]['distance'].sum()
            else:
                week_dist = 0
            week_totals.append(week_dist)
        ax.set_yticklabels([f"Week {i+1}\n{week_totals[i]:.1f} mi" for i in range(4)])

        # Map each date to the grid and plot activities
        for idx, day in enumerate(all_dates):
            week = idx // 7
            day_of_week = idx % 7
            cell_to_date[(day_of_week, week)] = day
            
            # Add the date label
            ax.text(day_of_week, week+datePosition, pd.Timestamp(day).strftime('%b %d'),
                    ha='center', va='top', fontsize=10, color='black', fontweight='bold')
            
            # Get all activities for this day
            day_acts = df_all_period[df_all_period['date'] == day] if not df_all_period.empty else pd.DataFrame()
            n_acts = len(day_acts)
            if n_acts == 0:
                continue
            
            # Calculate offsets for multiple activities
            if n_acts==1:
                 offsets = np.linspace(day_of_week, day_of_week + 0.2, n_acts)
            else:
                offsets = np.linspace(day_of_week - 0.2, day_of_week + 0.2, n_acts)
            
            # Plot each activity
            for k, (_, row) in enumerate(day_acts.iterrows()):
                activity_type = row['activity_type']
                add_activity_circle(ax, offsets[k], week, row, activity_type, config, trainingEffectLength)
            
        # Draw grid
        for x in range(8):
            ax.axvline(x-0.5, color='gray', lw=0.5)
        for y in range(5):
            ax.axhline(y-0.5, color='gray', lw=0.5)

        ax.set_title(f'Activities: Last 28 Days Ending {end_date.strftime("%Y-%m-%d")}')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout()
        
    return cell_to_date


def plotSplits(df_running_splits, df_running, ax, activityId=None, config=None):
    """
    Plots splits from the latest run.
    """
    if df_running_splits.empty or df_running.empty:
        print("No splits or running data available.")
        return
    
    if activityId is None:
        n = 0
        activity_id = int(df_running.iloc[n]['activityId'])
    else:
        activity_id = activityId
        
    print(f"Latest activity ID: {activity_id}")
    latest_splits = df_running_splits[df_running_splits['activityId'].astype(str) == str(activity_id)]

    # Sort by split number if needed (assumes 'lapIndex' or similar exists)
    latest_splits = latest_splits.sort_values(by='lapIndex') if 'lapIndex' in latest_splits else latest_splits
    if len(latest_splits) == 0:
        print(f"No splits found for activity ID {activity_id}.")
        return
    
    # Extract duration (sec) and distance (miles)
    durations = latest_splits['duration']  # seconds
    distances = latest_splits['distance'] * 0.00062137  # convert meters to miles if needed

    # Plot
    #import matplotlib.pyplot as plt
    #import numpy as np
    #fig, ax = plt.subplots(figsize=(10, 4))
    bar_widths = distances / distances.max()  # scale widths for visual clarity

    # Compute center positions for variable-width bars
    lefts = np.cumsum(np.insert(bar_widths[:-1], 0, 0))
    pace =  (durations/60) / distances
    #ax.bar(lefts, pace, width=bar_widths, align='edge', color='skyblue', edgecolor='k',bottom=10)

    threshold = float(config.get("plotting", {}).get("split_threshold", 7.0))

    colors = ['tomato' if p < threshold else 'skyblue' for p in pace]

    min = float(config.get("plotting", {}).get("split_ylim_min", 4.0))
    max = float(config.get("plotting", {}).get("split_ylim_max", 10.0))

    bars = ax.bar(lefts, max - pace, width=bar_widths, bottom=pace,
                align='edge', color=colors, edgecolor='k')
    
    def format_pace(p):
        minutes = int(p)
        seconds = int(round((p - minutes) * 60))
        return f"{minutes}:{seconds:02d}"

    # Example usage:
    formatted_paces = [format_pace(p) for p in pace]
    formatted_hr = [str(int(h)) for h in latest_splits['averageHR'].values]
    
    formatted_paces_hr = [f"{pace_str}  ({hr_str})" for pace_str, hr_str in zip(formatted_paces, formatted_hr)]  
    for left, p, label, width, hr in zip(lefts, pace, formatted_paces_hr, bar_widths, formatted_hr):
        ax.text(left + width / 2, p+.25, label,
            ha='center', va='top', rotation=90, fontsize=9)


    ax.invert_yaxis()

    distance = str(np.round(df_running[df_running['activityId'] == activity_id]['distance'].values[0],1))
    Pace = str(np.round(df_running[df_running['activityId'] == activity_id]['Avg Pace'].values[0],1))
    time = (df_running[df_running['activityId'] == activity_id]['duration_str'].values[0])
    elevationGain = str(np.round(df_running[df_running['activityId'] == activity_id]['elevationGain'].values[0]*3.28,1))
    metricStr = time + "\n" + distance + " mi\n" + Pace + " min/mi\n" + elevationGain + " ft"

    


    #ax.set_ylim(max, min)  # Keep this so your axis still reflects your preferred orientation
    print(f"Setting y-axis limits to {min} - {max}")

    ax.set_ylim(min, max)  # Keep this so your axis still reflects your preferred orientation

    ax.text(0.01,0.99, metricStr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',bbox=dict(boxstyle='round,pad=0.3',facecolor='wheat', alpha=0.85))
    ax.set_xlabel('Distance (miles)')
    ax.set_ylabel('Time (min/mile)')
    activityName = df_running[df_running['activityId'] == activity_id]['activityName'].astype(str).values[0]
  
    ax.set_title('Splits from Activity {}'.format(activityName))
    ax.invert_yaxis()
    plt.tight_layout()
    

    #print("Splits synced with df_summary with length = {} records".format(len(df_running_splits['activity_id'].unique())))
    #plt.show()

def plot_TSS(df,df_daily,ax):
    """
    Plots TSS (Training Stress Score) over time.
    """
    #plt.figure(figsize=(12, 6))
    ax.plot(df_daily['startTimeLocal'], df_daily['ctl'], label='CTL Chronic Training Load', linewidth=2)
    ax.plot(df_daily['startTimeLocal'], df_daily['atl'], label='ATL Acute Training Load', linewidth=2,alpha=0.5)
    ax.plot(df_daily['startTimeLocal'], df_daily['tsb'], label='TSB Training Stress Balance', linestyle='--', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_title('Training Load and Form')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    latest_date = df.index.max()
    earliest_date = latest_date - pd.DateOffset(years=10)
    # Set x-axis limits for all subplots
    ax.set_xlim([earliest_date, latest_date])
