"""Data splits."""

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.target_selection import leave_first, leave_last, leave_random


class LeaveOneOutSplitter:
    """
    A splitter that implements leave-one-out data splitting.

    This splitter divides the data into train, validation, and test sets where:
    - Train contains all but the last two interactions for each user
    - Validation contains all but the last interaction for each user (with the last in validation being the target)
    - Test contains all interactions with the very last interaction as the target

    The splitter can optionally filter out cold items (not seen in training) from validation and test sets.

    Args:
        user_col (str): Column name for user identifiers.
        item_col (str): Column name for item identifiers.
        timestamp_col (str): Column name for timestamps.
        remove_cold_items (bool, optional): Whether to remove items not present in training from validation/test.

    Returns:
        tuple: A tuple containing (train_data, validation_data, test_data), where each is a DataFrame or similar
            structure with the split data.

    The splitting process:
    1. Initial split into train, validation, and test
    2. For both validation and test sets:
        - The last interaction becomes the target
        - All previous interactions form the input sequence
    3. Cold items filtering (if remove_cold_items=True):
        - Removes items from validation/test targets that weren't seen in training
        - Removes corresponding input sequences that would predict cold items
    """

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
        remove_cold_items: bool = False,
    ) -> None:
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.remove_cold_items = remove_cold_items

    def _filter_cold_items(
        self,
        base: pd.DataFrame,
        df_input: pd.DataFrame,
        df_target: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters out cold items not present in base dataset.
        First, removes cold items from holdout and input, then filters out users with no target.

        Args:
            base (pd.DataFrame): Reference dataset (typically training)
            df_input (pd.DataFrame): Input sequences to filter
            df_target (pd.DataFrame): Target interactions to filter

        Returns:
            tuple: Filtered (df_input, df_target)
        """
        # Filter cold items from holdout
        warm_items = base[self.item_col].unique()
        df_target = df_target[df_target[self.item_col].isin(warm_items)]

        # Remove input sequences for holdout sequences where all items were removed
        users_with_targets = df_target[self.user_col].unique()
        df_input = df_input[
            df_input[self.user_col].isin(users_with_targets)
            & df_input[self.item_col].isin(warm_items)
        ]

        return df_input, df_target

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Performs the leave-one-out split on the input data.

        Args:
            data (pd.DataFrame): Interaction data to split, containing user_col and timestamp_col

        Returns:
            tuple: (train, validation_input, validation_target, test_input, test_target) where:
                - train: Training interactions
                - validation_input: Input sequences for validation
                - validation_target: Target interactions for validation
                - test_input: Input sequences for testing
                - test_target: Target interactions for testing
        """
        data = data.sort_values([self.user_col, self.timestamp_col], kind="stable")

        data["time_idx_reversed"] = data.groupby(self.user_col).cumcount(
            ascending=False
        )

        # Train contains all interactions except last two for each user
        train = data[data.time_idx_reversed >= 2].drop(columns=["time_idx_reversed"])

        # Validation contains all interactions except last one for each user
        validation = data[data.time_idx_reversed >= 1].drop(
            columns=["time_idx_reversed"]
        )

        # Split validation and test into input sequences and target (last user interaction within subset)
        validation_input, validation_target = leave_last(validation)
        test_input, test_target = leave_last(data.drop(columns=["time_idx_reversed"]))

        if self.remove_cold_items:
            # Remove items from validation/test that weren't seen in training
            validation_input, validation_target = self._filter_cold_items(
                train, validation_input, validation_target
            )

            test_input, test_target = self._filter_cold_items(
                train, test_input, test_target
            )

        return train, validation_input, validation_target, test_input, test_target


class GlobalTimeSplitter:
    """
    A temporal splitter that divides data based on global timestamps quantile.

    Splits data into training and test by a temporal quantile, with training being further split into training and validation by one of the following ways:
    - 'by_user': Random subset of users for validation
    - 'last_train_item': Last interaction per user in training as validation target
    - 'by_time': Additional temporal split within training data

    Optionally filters cold users/items and supports different target selection strategies.

    Args:
        quantile (float): Temporal quantile for train/test split (e.g., 0.9 = 90% earliest as train)
        validation_quantile (float): For 'by_time' validation, quantile within training.
        validation_type (str): Validation strategy ('by_user', 'last_train_item', 'by_time').
        validation_size (int): For 'by_user', number of users in validation.
        user_col (str): Column name for user identifiers.
        item_col (str): Column name for item identifiers.
        timestamp_col (str): Column name for timestamps.
        random_state (int): Random seed for reproducibility.
        remove_cold_users (bool): Filter users not in training from validation/test.
        remove_cold_items (bool): Filter items not in training from validation/test.
        target_type (str): How to select targets ('all', 'first', 'last', 'random').

    The splitting process:
    1. Initial split into train, validation, and test:
    - Train: All interactions before quantile (only sequences with ≥2 interactions)
    - Test: All user interactions with last interaction after quantile
    - Validation: Created via chosen strategy (by_user/last_train_item/by_time)

    2. For both validation and test sets:
    - Holdout period interactions become potential targets
    - All previous interactions form the input sequence

    3. Cold items filtering (if remove_cold_items=True):
    - Removes items from validation/test targets that weren't seen in training
    - Removes corresponding input sequences that would predict cold items

    4. Cold users filtering (if remove_cold_users=True):
    - Removes users from validation/test that weren't seen before (no input for given target)
    - Removes corresponding target sequences

    5. Target selection:
    - For 'val_by_user' and 'last_train_item': the last item in a sequence becomes target
    - For test and 'val_by_time': selects specified targets (first/last/random/all) from the holdout set
    - Combines interactions before target items with input sequences
    - Removes sequences with empty inputs
    """

    def __init__(
        self,
        quantile: float,
        validation_quantile: float = 0.9,
        validation_type: Literal["by_user", "last_train_item", "by_time"] = "by_user",
        validation_size: Optional[int] = 500,
        user_col: str = "user_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
        random_state: Optional[int] = 42,
        remove_cold_users: bool = False,
        remove_cold_items: bool = False,
        target_type: Literal["all", "first", "last", "random"] = "all",
    ) -> None:
        self.quantile = quantile
        self.validation_quantile = validation_quantile
        self.validation_type = validation_type
        self.validation_size = validation_size
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.random_state = random_state
        self.remove_cold_users = remove_cold_users
        self.remove_cold_items = remove_cold_items
        self.target_type = target_type

        np.random.seed(self.random_state)

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Performs the temporal split with validation according to configuration.

        Args:
            data (pd.DataFrame): Interaction data to split

        Returns:
            tuple: (train, validation_input, validation_target, test_input, test_target) where:
                - train: Training interactions
                - validation_input: Input sequences for validation
                - validation_target: Target interactions for validation
                - test_input: Input sequences for testing
                - test_target: Target interactions for testing
        """
        # Split into train and test by global time threshold
        train, test_input, test_holdout = self.split_by_time(data, self.quantile)

        # Create validation set according to specified strategy
        if self.validation_type == "by_user":
            train, validation_input, validation_target = self.split_validation_by_user(
                train
            )
        elif self.validation_type == "last_train_item":
            train, validation_input, validation_target = (
                self.split_validation_last_train(train)
            )
        elif self.validation_type == "by_time":
            train, validation_input, validation_target = self.split_by_time(
                train, self.validation_quantile
            )
        else:
            raise ValueError("Wrong validation_type.")

        # Handle cold start filtering
        validation_input, validation_target, test_input, test_holdout = (
            self._process_cold_entities(
                train, validation_input, validation_target, test_input, test_holdout
            )
        )

        # Process targets according to target_type (first/last/random/all)
        if self.validation_type == "by_time":
            validation_input, validation_target = self._process_target_type(
                validation_input, validation_target
            )
        test_input, test_target = self._process_target_type(test_input, test_holdout)

        if self.remove_cold_users:
            # Remove targets for zero-length inputs (e.g., target_type='last' when user had only one interaction)
            validation_input, validation_target = self._filter_cold_users(
                validation_input, validation_target
            )
            test_input, test_target = self._filter_cold_users(test_input, test_target)

        return train, validation_input, validation_target, test_input, test_target

    def split_by_time(
        self, data: pd.DataFrame, quantile: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data into train and test by a temporal quantile.

        Args:
            data (pd.DataFrame): Data to split
            quantile (float): Temporal quantile threshold

        Returns:
            tuple: (train, test_input, test_target) where:
                - train: Interactions before quantile
                - test_input: Input sequences before quantile for test users
                - test_target: Target interactions after quantile for test users
        """

        data = data.sort_values([self.user_col, self.timestamp_col], kind="stable")

        time_threshold = data[self.timestamp_col].quantile(quantile)

        # We need at least two items in a train sequence for training
        user_second_timestamp = data.groupby(self.user_col)[self.timestamp_col].nth(1)
        train_users = user_second_timestamp[
            user_second_timestamp <= time_threshold
        ].index
        train = data[data[self.user_col].isin(train_users)]

        # Train contains all interactions before the time threshold
        train = train[train[self.timestamp_col] <= time_threshold]

        # Test contains users with the last interaction after the time threshold
        user_last_timestamp = data.groupby(self.user_col)[self.timestamp_col].nth(-1)
        test_users = user_last_timestamp[user_last_timestamp > time_threshold].index

        test = data[data[self.user_col].isin(test_users)]

        test_input = test[test[self.timestamp_col] <= time_threshold]
        test_holdout = test[test[self.timestamp_col] > time_threshold]

        return train, test_input, test_holdout

    def split_validation_by_user(
        self,
        train: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Creates validation set by randomly selecting users from training.

        Args:
            train (pd.DataFrame): Training data

        Returns:
            tuple: (train, validation_input, validation_target) where validation contains random users
        """

        if self.validation_size is None:
            raise ValueError(
                "You must specify split_params.validation_size parameter for by_user splitting"
            )
        np.random.seed(self.random_state)

        # Randomly select validation users
        validation_users = np.random.choice(
            train[self.user_col].unique(), size=self.validation_size, replace=False
        )
        validation = train[train[self.user_col].isin(validation_users)]

        # Use last interaction for selected users as validation target
        validation_input, validation_target = leave_last(validation)

        train = train[~train[self.user_col].isin(validation_users)]

        return train, validation_input, validation_target

    def split_validation_last_train(
        self,
        train: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Creates validation set using last interaction per user in training as target.

        Args:
            train (pd.DataFrame): Training data

        Returns:
            tuple: (train, validation_input, validation_target) where validation targets are last train items
        """
        train = train.sort_values([self.user_col, self.timestamp_col], kind="stable")
        train["time_idx_reversed"] = train.groupby(self.user_col).cumcount(
            ascending=False
        )

        # Validation includes users with at least 2 interactions
        validation = train[
            train.groupby(self.user_col)["time_idx_reversed"].transform(max) > 0
        ].drop(columns=["time_idx_reversed"])
        # Use last interaction as validation target
        validation_input, validation_target = leave_last(validation)

        # Training sequences now exclude the validation target
        train = train[train.time_idx_reversed >= 1]

        # Keep only users with at least 2 interactions after validation split
        train = train[
            train.groupby(self.user_col)["time_idx_reversed"].transform(max) > 1
        ].drop(columns=["time_idx_reversed"])

        return train, validation_input, validation_target

    def _process_cold_entities(
        self,
        train: pd.DataFrame,
        validation_input: pd.DataFrame,
        validation_target: pd.DataFrame,
        test_input: pd.DataFrame,
        test_target: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Handles cold user/item filtering based on configuration.

        Args:
            train: Training data
            validation_input: Validation input sequences
            validation_target: Validation targets
            test_input: Test input sequences
            test_target: Test targets

        Returns:
            tuple: Filtered (validation_input, validation_target, test_input, test_target)
        """
        if self.remove_cold_items:
            validation_input, validation_target = self._filter_cold_items(
                train, validation_input, validation_target
            )
            test_input, test_target = self._filter_cold_items(
                train, test_input, test_target
            )

        if self.remove_cold_users:
            validation_input, validation_target = self._filter_cold_users(
                validation_input, validation_target
            )

            test_input, test_target = self._filter_cold_users(test_input, test_target)
        return validation_input, validation_target, test_input, test_target

    def _filter_cold_items(
        self,
        base: pd.DataFrame,
        df_input: pd.DataFrame,
        df_target: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters out cold items not present in base dataset.
        First, removes cold items from holdout and input, then filters out users with no target.

        Args:
            base (pd.DataFrame): Reference dataset (typically training)
            df_input (pd.DataFrame): Input sequences to filter
            df_target (pd.DataFrame): Target interactions to filter

        Returns:
            tuple: Filtered (df_input, df_target)
        """
        # Filter cold items from holdout
        warm_items = base[self.item_col].unique()
        df_target = df_target[df_target[self.item_col].isin(warm_items)]

        # Remove input sequences for holdout sequences where all items were removed
        users_with_targets = df_target[self.user_col].unique()
        df_input = df_input[
            df_input[self.user_col].isin(users_with_targets)
            & df_input[self.item_col].isin(warm_items)
        ]

        return df_input, df_target

    def _filter_cold_users(
        self,
        df_input: pd.DataFrame,
        df_target: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters out cold users with no input.

        Args:
            train (pd.DataFrame): Training data
            df_input (pd.DataFrame): Input sequences to filter
            df_target (pd.DataFrame): Target interactions to filter

        Returns:
            tuple: Filtered (df_input, df_target)
        """
        # Users must have both input and target sequences
        warm_users = df_input[self.user_col].unique()
        df_target = df_target[df_target[self.user_col].isin(warm_users)]

        return df_input, df_target

    def _process_target_type(
        self,
        input_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes targets according to target_type configuration.

        Args:
            holdout_data: Original holdout interactions
            input_data: Input sequences

        Returns:
            tuple: (input_data, target_data) processed according to target_type
        """
        if self.target_type == "all":
            return input_data, holdout_data

        dispatch = {
            "first": lambda: leave_first(holdout_data, input_data),
            "last": lambda: leave_last(holdout_data, input_data),
            "random": lambda: leave_random(holdout_data, input_data),
        }

        input_data, target_data = dispatch[self.target_type]()

        return input_data, target_data
