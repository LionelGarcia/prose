import os
from os import path
from pathlib import Path
import yaml
from yaml import Loader
import numpy as np
import shutil
from prose.telescope import built_in_telescopes
import glob

package_name = "prose"


# duplicate in io.py
def get_files(
    ext,
    folder,
    deepness=0,
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
    deepness : int
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
    for deepness in range(deepness + 1):
        files += glob.iglob(
            path.join(folder, "*/" * deepness + "*{}".format(ext)), recursive=False
        )

    files = [path.abspath(f) for f in files]

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

        home = str(Path.home())
        self.folder_path = path.abspath("{}/.{}".format(home, package_name))
        self.config_path = path.abspath("{}/.{}/config".format(str(Path.home()), package_name))

        self.check_config_file(load=True)

    def check_config_folder(self):
        if not path.exists(self.folder_path):
            os.mkdir(self.folder_path)
            print("{} config folder missing. Has been created!".format(self.folder_path))

    def check_config_file(self, load=False):

        self.check_config_folder()

        if path.exists(self.config_path):
            with open(self.config_path, "r") as file:
                if load:
                    self.config = yaml.load(file.read(), Loader=Loader)
        else:
            print("A config file as been created in {}".format(self.folder_path))
            self.config = {"color": "blue"}
            with open(self.config_path, "w") as file:
                yaml.dump(self.config, file, default_flow_style=False)

    def save(self):
        self.check_config_file()
        with open(self.config_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self.config[key] = value
        self.save()

    def check_telescope_files(self, name):
        id_files = get_files(".id", self.folder_path)

        if len(id_files) > 0:
            names = [file.strip(".id").lower for file in id_files]
            name_in_names = [name in n for n in names]

            if any(name_in_names):
                assert (
                    name_in_names.count == 1
                ), "Multiple .id files got matched with name {}".format(name)
                return id_files[np.where(name_in_names)[0]]

            else:
                return None

        else:
            return None

    def telescopes_dict(self):
        id_files = get_files(".id", self.folder_path, single_list_removal=False)

        _telescope_dict = {}

        for id_file in id_files:
            with open(id_file, "r") as f:
                telescope = yaml.load(f, Loader=yaml.FullLoader)
                _telescope_dict[telescope["name"].lower()] = telescope

        _telescope_dict.update(built_in_telescopes)

        return _telescope_dict

    def save_telescope_file(self, file):
        shutil.copyfile(file, path.join(self.folder_path, path.basename(file)))

    def match_telescope_name(self, name):
        if not isinstance(name, str):
            return None
        available_telescopes_names = list(self.telescopes_dict().keys())
        has_telescope = np.where(
            [t in name.lower() for t in available_telescopes_names]
        )[0]
        if len(has_telescope) > 0:
            return self.telescopes_dict()[available_telescopes_names[has_telescope[0]]]