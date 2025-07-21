"""
Filename: plots.py
Description: Plotting functions for Garmin Activity Tracker - visualizations of activity data.
Author: Ryan Kari
License: MIT
Created: 2025-07-20
"""
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

def plot_race_performance(df_summary,ax):
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

    ax.title('Race Outcomes')
    ax.grid(True)
    ax.set_ylabel('min/mi')
    ax.set_xlabel('Event Index')
    ax.set_ylim([6, 9.5])
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


def plot_calendar(dfRunning, ax, year=None, month=None):
    """
    Plots a calendar-style grid for a selected month (or last 4 weeks if no month/year given).
    Each activity is a circle sized by distance, races in red, others in skyblue.
    Handles months with more than 28 days.
    """
    datePosition = -0.32
    trainingEffectLength = 12
    cell_to_date = {}  # (x, y) -> date
    if year is not None and month is not None:
        # Month calendar grid
        first_day = pd.Timestamp(year=year, month=month, day=1)
        last_day = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
        df_month = dfRunning[(dfRunning.index >= first_day) & (dfRunning.index < last_day + pd.Timedelta(days=1))].copy()
        df_month['date'] = df_month.index.date

        # Use calendar.monthcalendar to get the grid (weeks x 7 days)
        month_calendar = calendar.monthcalendar(year, month)
        n_weeks = len(month_calendar)
        #fig, ax = plt.subplots(figsize=(14, 2.5 * n_weeks))
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, n_weeks - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(7))
        ax.set_yticks(range(n_weeks))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        # Calculate total distance for each week
        week_totals = []
        for i, week in enumerate(month_calendar):
            week_dates = [pd.Timestamp(year=year, month=month, day=day).date()
                            for day in week if day != 0]
            week_dist = df_month[df_month['date'].isin(week_dates)]['distance'].sum()
            week_totals.append(week_dist)
        ax.set_yticklabels([f"Week {i+1}\n{week_totals[i]:.1f} mi" for i in range(n_weeks)])

        # Map each date to (week, day)
        for i, week in enumerate(month_calendar):
            for j, day in enumerate(week):
                if day == 0:
                    continue  # No day in this cell
                day_date = pd.Timestamp(year=year, month=month, day=day).date()
                cell_to_date[(j, i)] = day_date
                # Add the date label in the cell
                ax.text(j, i+datePosition, pd.Timestamp(day_date).strftime('%b %d'),
                        ha='center', va='top', fontsize=10, color='black', fontweight='bold')
                day_acts = df_month[df_month['date'] == day_date]
                n_acts = len(day_acts)
                if n_acts == 0:
                    continue
                offsets = np.linspace(0, 0.2, n_acts)
                for k, (_, row) in enumerate(day_acts.iterrows()):
                    size = distance_to_size(row['distance'])
                    color = 'red' if str(row.get('Race', '')).lower() == 'race' else 'skyblue'
                    ax.scatter(j + offsets[k], i, s=size, color=color, edgecolor='k', alpha=0.7)
                    ax.text(j + offsets[k], i, f"{row['distance']:.1f}", ha='center', va='center', fontsize=8)
                trainingEffectLabel = row["trainingEffectLabel"]
                if isinstance(trainingEffectLabel, str):
                    ax.text(j + offsets[k], i+.25, trainingEffectLabel[0:trainingEffectLength],
                            ha='center', va='top', fontsize=8, color='black')
        # Draw grid
        for x in range(8):
            ax.axvline(x-0.5, color='gray', lw=0.5)
        for y in range(n_weeks + 1):
            ax.axhline(y-0.5, color='gray', lw=0.5)

        ax.set_title(f'Running Activities: {calendar.month_name[month]} {year} (Calendar View)')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout()
        

    else:
        # Default: last 28 days ending with the most recent activity (as before)
        #end_date = dfRunning.index.max().normalize()
        end_date = pd.Timestamp(pd.Timestamp.now().date())
        start_date = end_date - pd.Timedelta(days=27)
        df_4w = dfRunning[(dfRunning.index >= start_date) & (dfRunning.index < end_date + pd.Timedelta(days=1))].copy()
        df_4w['date'] = df_4w.index.date
        all_dates = [end_date - pd.Timedelta(days=x) for x in reversed(range(28))]
        all_dates = [d.date() for d in all_dates]
        n_weeks = 4
        #fig, ax = plt.subplots(figsize=(14, 2.5 * n_weeks))
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, n_weeks - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(7))
        ax.set_yticks(range(4))
        weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        first_weekday = pd.Timestamp(all_dates[0]).weekday()
        rotated_labels = weekday_labels[first_weekday:] + weekday_labels[:first_weekday]
        ax.set_xticklabels(rotated_labels)

        week_totals = []
        for week in range(4):
            week_start_idx = week * 7
            week_dates = all_dates[week_start_idx:week_start_idx + 7]
            week_dist = df_4w[df_4w['date'].isin(week_dates)]['distance'].sum()
            week_totals.append(week_dist)
        ax.set_yticklabels([f"Week {i+1}\n{week_totals[i]:.1f} mi" for i in range(4)])

        for idx, day in enumerate(all_dates):
            week = idx // 7
            day_of_week = idx % 7
            cell_to_date[(day_of_week, week)] = day
            ax.text(day_of_week, week+datePosition, pd.Timestamp(day).strftime('%b %d'),
                    ha='center', va='top', fontsize=10, color='black', fontweight='bold')
            day_acts = df_4w[df_4w['date'] == day]
            n_acts = len(day_acts)
            if n_acts == 0:
                continue
            offsets = np.linspace(0, 0.2, n_acts)
            for k, (_, row) in enumerate(day_acts.iterrows()):
                size = distance_to_size(row['distance'])
                color = 'red' if str(row.get('Race', '')).lower() == 'race' else 'skyblue'
                ax.scatter(day_of_week + offsets[k], week, s=size, color=color, edgecolor='k', alpha=0.7)
                ax.text(day_of_week + offsets[k], week, f"{row['distance']:.1f}", ha='center', va='center', fontsize=8)
            
            ax.text(day_of_week + offsets[k], week+.25, row["trainingEffectLabel"][0:trainingEffectLength],
                    ha='center', va='top', fontsize=8, color='black')
            
        for x in range(8):
            ax.axvline(x-0.5, color='gray', lw=0.5)
        for y in range(5):
            ax.axhline(y-0.5, color='gray', lw=0.5)

        ax.set_title(f'Running Activities: Ending {end_date.strftime("%Y-%m-%d")}')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout()
    return cell_to_date


def plotSplits(df_splits, df_running, ax, activityId=None):
    """
    Plots splits from the latest run.
    """
    if df_splits.empty or df_running.empty:
        print("No splits or running data available.")
        return
    
    if activityId is None:
        n = 0
        activity_id = int(df_running.iloc[n]['activityId'])
    else:
        activity_id = activityId
        
    print(f"Latest activity ID: {activity_id}")
    latest_splits = df_splits[df_splits['activityId'] == activity_id]

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

    colors = ['tomato' if p < 7 else 'skyblue' for p in pace]



    bars = ax.bar(lefts, 10 - pace, width=bar_widths, bottom=pace,
                align='edge', color=colors, edgecolor='k')
    
    def format_pace(p):
        minutes = int(p)
        seconds = int(round((p - minutes) * 60))
        return f"{minutes}:{seconds:02d}"

    # Example usage:
    formatted_paces = [format_pace(p) for p in pace]

    for left, p, label, width in zip(lefts, pace, formatted_paces, bar_widths):
        ax.text(left + width / 2, p+.25, label,
            ha='center', va='top', rotation=90, fontsize=9)

    ax.invert_yaxis()
    ax.set_ylim(4, 10)  # Keep this so your axis still reflects your preferred orientation

    ax.set_xlabel('Distance (miles)')
    ax.set_ylabel('Time (min/mile)')
    activityName = df_running[df_running['activityId'] == activity_id]['activityName'].astype(str).values[0]
  
    ax.set_title('Splits from Activity {}'.format(activityName))
    ax.invert_yaxis()
    #ax.set_ylim(10, 4)  
    plt.tight_layout()
    

    #print("Splits synced with df_summary with length = {} records".format(len(df_splits['activity_id'].unique())))
    #plt.show()

def plot_TSS(df_daily,ax):
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
    #ax.tight_layout()
    #plt.show()