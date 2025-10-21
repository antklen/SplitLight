"""
Preprocessing utils.
"""


def encode(data, col_name, shift):
    """Encode items/users to consecutive ids.

    :param col_name: column to do label encoding, e.g. 'item_id'
    :param shift: shift encoded values to start from shift
    """
    data[col_name] = data[col_name].astype("category").cat.codes + shift
    return data


def rename_cols(data, user_id="user_id", item_id="item_id", timestamp="timestamp"):
    "Rename columns of dataframe"

    data = data.rename(
        columns={user_id: "user_id", item_id: "item_id", timestamp: "timestamp"}
    )

    return data
