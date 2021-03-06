from os import path
from pathlib import Path
import yaml
from yaml import Loader
import numpy as np
import shutil
from .builtins import built_in_telescopes
import glob
import requests

info = print
package_name = "prose"


# duplicate in io.py
def get_files(
    ext,
    folder,
    depth=0,
    return_folders=False,
    single_list_removal=True,
    none_for_empty=False,
):
    """

    Return files of specific extension in the specified folder and sub-folders

    Parameters
    ----------
    folder : str
        Folder to be analyzed
    depth : int
        Number how sub-folder layer to look into.
        0 (default) will look into current folder
        1 will look into current folder and its sub-folders
        2 will look into current folder, its sub-folders and their sub-folders
        ... etc

    Returns
    -------
    list of fits files

    """
    files = []
    for depth in range(depth + 1):
        files += glob.iglob(
            path.join(folder, "*/" * depth + "*{}".format(ext)), recursive=False
        )

    files = [path.abspath(f) for f in files if path.isfile(f)]

    if return_folders:
        folders = [path.dirname(file) for file in files]
        if single_list_removal and len(folders) == 1:
            return folders[0]
        else:
            return folders
    else:
        if single_list_removal and len(files) == 1:
            return files[0]
        elif len(files) == 0 and none_for_empty:
            return None
        else:
            return files


class ConfigManager:
    def __init__(self):

        self.config = None

        self.folder_path = Path.home() / f".{package_name}"
        self.folder_path.mkdir(exist_ok=True)

        self.config_file = self.folder_path / "config"

        self.check_config_file(load=True)
        self.telescopes_dict = self.build_telescopes_dict()
        self.check_ballet()

    def check_config_file(self, load=False):

        if self.config_file.exists():
            with self.config_file.open(mode="r") as file:
                if load:
                    self.config = yaml.load(file.read(), Loader=Loader)
        else:
            info(f"A config file as been created in {self.folder_path}")
            self.config = {"color": "blue"}
            with self.config_file.open(mode="w") as file:
                yaml.dump(self.config, file, default_flow_style=False)

    def save(self):
        self.check_config_file()
        with self.config_file.open(mode="w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self.config[key] = value
        self.save()

    def build_telescopes_dict(self):
        id_files = self.folder_path.glob("*id")

        telescope_dict = {}

        for id_file in id_files:
            telescope = yaml.load(id_file.open(mode="r"), Loader=yaml.FullLoader)
            telescope_dict[telescope["name"].lower()] = telescope
            if hasattr(telescope, "names"):
                for name in telescope["names"]:
                    telescope_dict[name.lower()] = telescope

        telescope_dict.update(built_in_telescopes)

        return telescope_dict

    def save_telescope_file(self, file):
        if isinstance(file, str):
            name = Path(file).stem.lower()
            shutil.copyfile(file, self.folder_path / f"{name}.id")
            info("Telescope '{}' saved".format(name))
        elif isinstance(file, dict):
            name = file["name"].lower()
            telescope_file_path =  self.folder_path / f"{name}.id"
            yaml.dump(file, telescope_file_path.open(mode="w"))
            info("Telescope '{}' saved".format(name))
        else:
            raise AssertionError("input type not understood")
        self.telescopes_dict = self.build_telescopes_dict()

    def match_telescope_name(self, name):
        if not isinstance(name, str):
            return None
        available_telescopes_names = list(self.telescopes_dict.keys())
        has_telescope = np.where(
            [t in name.lower() for t in available_telescopes_names]
        )[0]
        if len(has_telescope) > 0:
            i = np.argmax([len(name) for name in np.array(available_telescopes_names)[has_telescope]])
            return self.telescopes_dict[available_telescopes_names[has_telescope[i]]]

    def check_ballet(self):
        model_path = self.folder_path / "centroid.h5"

        if not model_path.exists():
            print("downloading ballet model (~30Mb)")
            model = requests.get("https://github.com/lgrcia/ballet/raw/master/models/centroid.h5").content
            model_path.open(mode="wb").write(model)
