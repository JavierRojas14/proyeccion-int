def create_features_datetime_index(df):
    """
    Create datetime features based on datetime index
    """
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear

    return df
