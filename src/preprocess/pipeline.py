import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.filters import core_filter
from src.preprocess.utils import encode, rename_cols
from src.stats.base import base_stats


def preprocess(
    data,
    item_min_count=5,
    seq_min_len=5,
    core=True,
    encoding=True,
    drop_conseq_repeats=False,
    filter_by_relevance=False,
    users_sample=None,
    user_id="user_id",
    item_id="item_id",
    timestamp="timestamp",
    relevance="relevance",
    path_to_save=None,
    verbose=True,
):
    """
    - columns renaming
    - N-core or N-filter for items and sequences along with iterative
    - removal of consecutive interactions with the same item
    - label encoding of users and items, item labels starts with 1 to leave 0 as a padding value
    """
    # os.chdir('../')

    # filter columns TO DO: make optional
    columns = [
        key for key in [user_id, item_id, timestamp, relevance] if key is not None
    ]

    data = data[columns].copy()

    data = rename_cols(data, user_id, item_id, timestamp)

    if verbose:
        print("Raw data")
        print(base_stats(data))

    if filter_by_relevance:
        raise NotImplementedError("No filter_by_relevance implemented")

    if users_sample is not None:
        raise NotImplementedError("No user sampling implemented")

    if core:
        data = core_filter(
            data=data,
            item_min_count=item_min_count,
            seq_min_len=seq_min_len,
            drop_conseq_repeats=drop_conseq_repeats,
            user_id="user_id",
            item_id="item_id",
            timestamp="timestamp",
            verbose=verbose,
        )

        if verbose:
            print("After N-core")
            print(base_stats(data))

    else:
        raise NotImplementedError("N-core filtering is only one available")

    if encoding:
        data = encode(data=data, col_name="user_id", shift=0)
        data = encode(data=data, col_name="item_id", shift=1)

    if path_to_save is not None:
        data.to_csv(path_to_save, index=False)

    return data
