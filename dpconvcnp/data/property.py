from typing import List, Tuple, Optional, Literal
from copy import deepcopy

import numpy as np
import tensorflow as tf
import pandas as pd

from dpconvcnp.data.data import Batch, DataGenerator
from dpconvcnp.random import Seed, randint, to_tensor
from dpconvcnp.utils import f32


class PropertyPriceDataGenerator(DataGenerator):
    def __init__(
        self,
        *,
        path_to_csv: str,
        task_size: int,
        mode: Literal["train", "valid", "test"],
        valid_fraction: float = 0.1,
        test_fraction: float = 0.1,
        property_types: Optional[Tuple[str]] = None,
        lease_types: Optional[Tuple[str]] = None,
        age_types: Optional[Tuple[str]] = None,
        min_coords: Optional[Tuple[float]] = None,
        max_coords: Optional[Tuple[float]] = None,
        reset_seed_at_epoch_end: bool = False,
        **kwargs,
    ):
        # Set dataloader parameters
        self.mode = mode
        self.property_types = property_types
        self.lease_types = lease_types
        self.age_types = age_types
        self.min_coords = min_coords
        self.max_coords = max_coords
        self.valid_fraction = valid_fraction
        self.test_fraction = test_fraction
        self.task_size = task_size

        # Read data csv
        dataframe = pd.read_csv(path_to_csv)

        # Drop any rows with NaNs
        dataframe = dataframe.dropna(subset=["price", "lat", "lon", "day"])

        # Filter dataframe to keep properties that fit specified criteria
        dataframe = self.filter_dataframe(dataframe=dataframe)

        # Order dataframe by day
        dataframe = dataframe.sort_values(by="day", ascending=True)

        # Compute samples per epoch and remove any excess rows
        B = kwargs["batch_size"]
        dataframe = dataframe[
            : (dataframe.shape[0] // (B * task_size)) * B * task_size
        ]
        dataframe = self.zero_mean_price(dataframe=dataframe)
        self.dataframe = self.normalise_lat_lon(dataframe=dataframe)

        self.tasks = np.split(
            self.dataframe,
            len(self.dataframe) // task_size,
        )

        self.task_index = self.make_task_index()
        samples_per_epoch = len(self.task_index)

        self.reset_seed_at_epoch_end = reset_seed_at_epoch_end
        self.base_seed = deepcopy(kwargs["seed"])

        super().__init__(samples_per_epoch=samples_per_epoch, **kwargs)

    def __iter__(self):
        """Reset epoch counter and seed and return self."""
        self.seed = (
            self.base_seed if self.reset_seed_at_epoch_end else self.seed
        )
        return super().__iter__()

    def filter_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.property_types is not None:
            dataframe = dataframe[dataframe["type"].isin(self.property_types)]

        if self.lease_types is not None:
            dataframe = dataframe[dataframe["lease"].isin(self.lease_types)]

        if self.age_types is not None:
            dataframe = dataframe[dataframe["new"].isin(self.age_types)]

        if self.min_coords is not None:
            dataframe = dataframe[dataframe["lat"] >= self.min_coords[0]]
            dataframe = dataframe[dataframe["lon"] >= self.min_coords[1]]

        if self.max_coords is not None:
            dataframe = dataframe[dataframe["lat"] <= self.max_coords[0]]
            dataframe = dataframe[dataframe["lon"] <= self.max_coords[1]]

        return dataframe

    def make_task_index(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Create train, valid and test indices
        split = int(len(self.tasks) * (1 - self.test_fraction))
        train_valid_idx = np.arange(split)

        # Split train into train and valid
        n = int((1.0 - self.test_fraction) / self.valid_fraction)

        if self.mode == "train":
            return train_valid_idx[np.arange(len(train_valid_idx)) % n != 0]

        elif self.mode == "valid":
            return train_valid_idx[np.arange(len(train_valid_idx)) % n == 0]

        elif self.mode == "test":
            return np.arange(split, len(self.tasks))

        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def normalise_lat_lon(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        max_lat = dataframe["lat"].max()
        min_lat = dataframe["lat"].min()
        mid_lat = (max_lat + min_lat) / 2.0
        scale_lat = (max_lat - min_lat) / 2.0

        max_lon = dataframe["lon"].max()
        min_lon = dataframe["lon"].min()
        mid_lon = (max_lon + min_lon) / 2.0
        scale_lon = (max_lon - min_lon) / 2.0

        dataframe["lat"] = (dataframe["lat"] - mid_lat) / scale_lat
        dataframe["lon"] = (dataframe["lon"] - mid_lon) / scale_lon

        return dataframe

    def zero_mean_price(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["price"] = dataframe["price"] - dataframe["price"].mean()
        return dataframe

    def generate_data(self, seed: Seed) -> Tuple[Seed, Batch]:
        seed, idx = randint(
            seed=seed,
            shape=(self.batch_size,),
            minval=0,
            maxval=len(self.tasks) - 1,
        )

        # Get tasks for batch, using the .sample function to shuffle
        tasks = [self.tasks[i].sample(frac=1.0) for i in idx.numpy()]

        lat = np.stack([task["lat"].values for task in tasks], axis=0)
        lon = np.stack([task["lon"].values for task in tasks], axis=0)
        price = np.stack([task["price"].values for task in tasks], axis=0)

        x = to_tensor(np.stack([lat, lon], axis=-1), f32)
        y = to_tensor(price, f32)[:, :, None]

        # Split into context and target
        seed, num_ctx = randint(
            seed=seed,
            shape=(),
            minval=1,
            maxval=self.task_size - 1,
        )

        return seed, Batch(
            x=x,
            y=y,
            x_ctx=x[:, :num_ctx, :],
            y_ctx=y[:, :num_ctx, :],
            x_trg=x[:, num_ctx:, :],
            y_trg=y[:, num_ctx:, :],
        )