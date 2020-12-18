import shutil
from os import path
import pandas as pd
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from ..telescope import Telescope
from datetime import timedelta
import os
from .io import get_files, fits_to_df


class FilesDataFrame:
    def __init__(self, files_df, verbose=True):
        self.files_df = None
        self.verbose = verbose

        self.files_df = files_df.reset_index(drop=True)
        self._original_files_df = self.files_df.copy()

        assert len(self._original_files_df) != 0, "No data found"
        self.telescope = None

    def reset(self):
        self.files_df = self._original_files_df.copy()

    #     def __getattribute__(self)

    @staticmethod
    def _get(files_df, return_conditions=False, **kwargs):
        conditions = pd.Series(np.ones(len(files_df)).astype(bool))

        for field, value in kwargs.items():
            #             if "*" not in value: value += "*"
            if isinstance(value, str):
                conditions = conditions & (
                    files_df[field].astype(str).str.lower().str.match(value.lower())).reset_index(drop=True)
            else:
                conditions = conditions & (files_df[field] == value).reset_index(drop=True)

        if return_conditions:
            return conditions
        else:
            return files_df.reset_index(drop=True).loc[conditions]

    def get(self, return_conditions=False, **kwargs):
        return self._get(self.files_df, return_conditions=False, **kwargs)

    def keep(self, inplace=True, **kwargs):

        new_df = self.get(**kwargs)
        if inplace:
            self.files_df = new_df
        else:
            return self.__class__(new_df)

        self.sort_by_date()

    def sort_by_date(self):
        if "jd" in self.files_df:
            self.sort_by("jd")

    def sort_by(self, field, inplace=True):
        if field in self.files_df:
            new_df = self.files_df.sort_values([field]).reset_index(drop=True)
            if inplace:
                self.files_df = new_df
            else:
                return new_df
        else:
            raise KeyError("'{}' is not in files_df".format(field))

    def describe(self, *fields, return_string=False, original=False, unique=True, index=False, **kwargs):

        if original:
            files_df = self._original_files_df.copy()
        else:
            files_df = self.files_df.copy()

        files_df = files_df.fillna("")

        if len(kwargs) > 0:
            files_df = self._get(files_df, **kwargs)
            headers = list(fields) + list(kwargs.keys())

        else:
            headers = list(fields)

        if unique:
            multi_index_obs = files_df.pivot_table(index=headers, aggfunc="size")
            if index:
                headers.insert(0, "index")
            single_index_obs = multi_index_obs.reset_index().rename(columns={0: 'quantity'}).reset_index(level=0)
            rows = OrderedDict(single_index_obs[[*headers, "quantity"]].to_dict(orient="list"))
            table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys")
        else:
            rows = OrderedDict(files_df[headers].to_dict(orient="list"))
            table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys")

        if return_string:
            return table_string
        else:
            print(table_string)

    def move_to(self, dirname):
        if not path.exists(dirname):
            os.mkdir(dirname)

        for file in self.files_df["path"].values:
            shutil.move(file, dirname)

    def copy_to(self, dirname):
        if not path.exists(dirname):
            os.mkdir(dirname)

        for file in self.files_df["path"].values:
            shutil.copy(file, dirname)

    def __repr__(self):
        return self.files_df.__repr__()

    def _repr_html_(self):
        return self.files_df._repr_html_()


class FitsManager(FilesDataFrame):

    def __init__(self, files_df_or_folder, verbose=True, image_kw="light", **kwargs):
        if isinstance(files_df_or_folder, pd.DataFrame):
            files_df = files_df_or_folder
            self.folder = None
        elif isinstance(files_df_or_folder, str):
            assert path.exists(files_df_or_folder), "Folder does not exist"
            files = get_files("*.f*ts", files_df_or_folder, depth=kwargs.get("depth", 1))
            files_df = fits_to_df(files)
            self.folder = files_df_or_folder
        else:
            raise AssertionError("input must be pd.DataFrame or folder path")

        super().__init__(files_df, verbose=verbose)
        self.image_kw = image_kw

    @property
    def observations(self):
        headers = ["date", "telescope", "target", "filter"]
        self.describe(*headers, index=True)
        return None

    @property
    def _observations(self):
        light_rows = self.files_df.loc[self.files_df["type"].str.contains(self.image_kw).fillna(False)]
        observations = (
            light_rows.pivot_table(
                index=["date", "telescope", "target", "dimensions", "filter"],
                aggfunc="size",
            )
                .reset_index()
                .rename(columns={0: "quantity"})
                .reset_index(level=0)
        )

        return observations

    @property
    def calib(self):
        headers = ["date", "telescope", "target", "filter", "type"]
        self.describe(*headers, index=False)
        return None

    def set_observation(self, i, **kwargs):
        self.files_df = self.get_observation(i, **kwargs)
        assert self.unique_obs, "observation should be unique, please use set_observation"
        obs = self._observations.loc[0]
        self.telescope = Telescope.from_name(obs.telescope)
        self.sort_by_date()

    def get_observation(self, i, future=0, past=None, same_telescope=False):

        original_fm = FitsManager(self._original_files_df.fillna(""))
        obs = self._observations.loc[i]

        days_limit = (1 if future is not None else -1)
        if future is not None:
            days_limit = -future
        elif past is not None:
            days_limit = past

        telescope = obs.telescope if same_telescope else "."
        dimensions = obs.dimensions

        dates_before = (
                    pd.to_datetime(obs.date) - pd.to_datetime(original_fm.files_df.date) >= timedelta(days=days_limit))

        flats = original_fm.get(
            telescope=telescope + "*", filter=obs["filter"].replace("+", "\+"),
            type="flat",
            dimensions=dimensions).loc[ dates_before]
        darks = original_fm.get(telescope=telescope + "*", type="dark", dimensions=dimensions).loc[dates_before]
        bias = original_fm.get(telescope=telescope + "*", type="bias", dimensions=dimensions).loc[dates_before]
        dfs = []

        if len(flats) > 0:
            flats = flats.loc[flats.date == flats.date.values[pd.to_datetime(flats.date).argmax()]]
            dfs.append(flats)
        if len(darks) > 0:
            darks = darks.loc[darks.date == darks.date.values[pd.to_datetime(darks.date).argmax()]]
            dfs.append(darks)
        if len(bias) > 0:
            bias = bias.loc[bias.date == bias.date.values[pd.to_datetime(bias.date).argmax()]]
            dfs.append(bias)

        others = original_fm.get(
            telescope=obs.telescope,
            filter=obs["filter"].replace("+", "\+"),
            target=obs.target.replace("+", "\+"),
            date=obs.date)

        for _type in ["dark", "flat", "bias"]:
            others = others.drop(np.argwhere(FilesDataFrame._get(others, type=_type, return_conditions=True).values))

        dfs.append(others)

        return pd.concat([pd.concat(dfs)])

    @property
    def unique_obs(self):
        return len(self._observations) == 1

    @property
    def images(self):
        return self.get(type=self.image_kw).path.values.astype(str)

    @property
    def darks(self):
        return self.get(type="dark").path.values.astype(str)

    @property
    def bias(self):
        return self.get(type="bias").path.values.astype(str)

    @property
    def flats(self):
        return self.get(type="flat").path.values.astype(str)

    @property
    def stack(self):
        return self.get(type="stack").path.values.astype(str)

    @property
    def products_denominator(self):
        assert self.unique_obs, "observation should be unique, please use set_observation"
        obs = self._observations.loc[0]
        return f"{obs.telescope}_{obs.date.replace('-', '')}_{obs.target}_{obs['filter']}"