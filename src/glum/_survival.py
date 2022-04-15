import pandas as pd


def _check_duration_column(df, duration_column):
    """Missings are allowed!"""
    if not pd.api.types.is_numeric_dtype(df[duration_column]):
        raise TypeError(f"Duration {duration_column} must be numeric.")
    if (df[duration_column] < 0).any():
        raise ValueError(f"Duration {duration_column} must not have negative values.")


def _check_event_column(df, event_column):
    if not pd.api.types.is_bool_dtype(df[event_column]):
        raise TypeError(f"Event indicator {event_column} must be boolean.")


def _check_times(times):
    # TBD if use
    if not all([t >= 0.0 for t in times]):
        raise TypeError("times must be positive.")
    if len(set(times)) != len(times):
        raise ValueError("Found duplicated values in times.")


# %% terminology
# duration: time until event or censoring
# event: event or censoring indicator
# time: endpoint of interval for which a separate baseline hazard (or: time-fixed effect) will be estimated
def survival_split(
    df, duration_column, times=None, event_column=None, time_column="time"
):
    _check_duration_column(df, duration_column)
    if event_column is not None:
        _check_event_column(df, event_column)
    if times is None:
        times = list(df[duration_column].dropna().drop_duplicates())
    else:
        _check_times(times)
    _df_times = (
        pd.DataFrame({time_column: times, "_key": 0})
        .sort_values(time_column)
        .reset_index(drop=True)
    )
    # %% outer merge to create data split:
    df["_key"] = 0
    df_split = df.merge(_df_times, on="_key", how="outer").drop(columns="_key")
    df_split = df_split.loc[
        df_split[duration_column] >= df_split[time_column]
    ].reset_index(drop=True)
    if event_column is not None:
        df_split.loc[
            df_split[duration_column] > df_split[time_column], event_column
        ] = False
    return df_split
