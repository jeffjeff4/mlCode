##import pandas as pd
##import numpy as np
##
### Create sample datetime data
##dates = pd.Series(pd.date_range('2023-01-01', periods=10, freq='12H'))  # 12-hour intervals
##
### Create DataFrame with extracted features
##dt_features = pd.DataFrame({
##    'original_datetime': dates,
##
##    # Date components
##    'year': dates.dt.year,
##    'month': dates.dt.month,
##    'day': dates.dt.day,
##
##    # Time components
##    'hour': dates.dt.hour,
##    'minute': dates.dt.minute,
##    'second': dates.dt.second,
##
##    # Week information
##    'weekday': dates.dt.weekday,  # Monday=0, Sunday=6
##    'weekday_name': dates.dt.day_name(),  # Full day name
##    'is_weekend': dates.dt.weekday >= 5,  # Saturday/Sunday
##
##    # Week of year
##    'week_of_year': dates.dt.isocalendar().week,
##
##    # Quarter
##    'quarter': dates.dt.quarter,
##
##    # Special day indicators
##    'is_month_start': dates.dt.is_month_start,
##    'is_month_end': dates.dt.is_month_end,
##
##    # Time of day categories
##    'time_of_day': pd.cut(dates.dt.hour,
##                          bins=[0, 6, 12, 18, 24],
##                          labels=['Night', 'Morning', 'Afternoon', 'Evening'],
##                          right=False)
##})
##
##print("dt_features")
##print(dt_features)
##print('\n')