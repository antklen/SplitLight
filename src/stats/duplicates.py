from typing import Dict, Optional

import pandas as pd

from .utils import get_conseq_duplicates


def get_item_repeat(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Adds a flag column indicating duplicate item interactions (regardless of order).
    
    Args:
        data: DataFrame containing user interactions with columns: user_id, item_id
        
    Returns:
        DataFrame with 'item_duplicate' column added
    """
    data['item_duplicate'] = data.duplicated(subset=['user_id', 'item_id'], keep='first')
    
    return data

def get_all_duplicates(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Adds all three duplicate flag columns to the DataFrame.
    
    Args:
        data: DataFrame containing user interactions
        
    Returns:
        DataFrame with all duplicate flag columns added
    """
    data = data.copy()
    
    # Add all three flags
    data = get_conseq_duplicates(data)
    data = get_item_repeat(data)
    
    return data

def _duplicate_counts(
    data: pd.DataFrame,
    col="item_duplicate",
    count_no_duplicates=False
) -> Dict[str, float]:
    """
    Calculates duplicate statistics in a consistent format.
    
    Args:
        data: DataFrame containing user interactions with columns: user_id, timestamp and duplicate flag column
        col: name of the duplicate column
        
    Returns:
        Dictionary with formatted statistics
    """
    grouped = data.groupby('user_id')[col]

    users = grouped.any()
    num_repeats = grouped.sum()
    share_repeats = grouped.mean()

    if count_no_duplicates:
        relevant_users = users.index
    else:
        relevant_users = users[users].index

    avg_number_per_user = num_repeats.loc[relevant_users].mean()
    avg_number_per_user = 0 if pd.isna(avg_number_per_user) else avg_number_per_user

    avg_share_per_user = share_repeats.loc[relevant_users].mean()
    avg_share_per_user = 0 if pd.isna(avg_share_per_user) else avg_share_per_user

    return {
        "Number of Users": users.sum(),
        "Share of Users": users.mean(),
        "Avg. Number per user": avg_number_per_user,
        "Avg. Share per user": avg_share_per_user,
    }

def duplicates_stats(data: pd.DataFrame, count_no_duplicates=False) -> pd.DataFrame:
    """
    Aggregates statistics for three types of duplicate interactions:
    - Consecutive item duplicates
    - Non-unique item interactions

    Args:
        data (pd.DataFrame): A DataFrame containing user interactions.

    Returns:
        pd.DataFrame: A DataFrame with each column representing a duplicate type, and stats as rows.
    """
    data_all_flags = get_all_duplicates(data)

    stats_dict = {
        "Consecutive item duplicates": _duplicate_counts(data_all_flags, "conseq_duplicate", count_no_duplicates),
        "Non-unique item interactions": _duplicate_counts(data_all_flags, "item_duplicate", count_no_duplicates),
    }

    return pd.DataFrame(stats_dict)
