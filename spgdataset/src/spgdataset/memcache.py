import os
import sys
from torch import Tensor, tensor, float32, load
import numpy as np
import pickle
from tqdm import tqdm
from fnmatch import fnmatch
from enum import Enum
import logging


class Status(Enum):
    OK = 1
    FULL = -1
    ALREADY_EXISTS = 0


class MemoryStorage:
    #
    # Stores data (torch.Tensor) in memory and provides methods to retrieve and add data.
    #
    def __init__(
        self, root: str, mask: str = "*", mode: str = "pickle", limit: int = 0
    ):
        """
        Initialize the MemoryStorage object.
        Args:
            root (str): The root directory where the files are located.
            mask (str, optional): The file mask to filter files. Defaults to "*".
            mode (str, optional): The mode for loading files, either 'pickle' or 'torch'. Defaults to "pickle".
            limit (int, optional): The memory limit for caching files, in MB. If set to 0, no files will be retained in memory. Defaults to 0.
        Raises:
            AssertionError: If the mode is not 'pickle' or 'torch'.
        """

        self.root = root
        self.mask = mask
        assert mode in ["pickle", "torch"], "Mode must be either 'pickle' or 'torch'"
        self.mode = mode
        self.limit = limit
        self.storage = {}
        self.running_size = 0
        if self.limit == 0:
            self.full = True
            logging.info(
                "[spgdataset/MEMCACHE] Memory limit is set to 0. No files will be retained in memory"
            )
        else:
            self.full = False
            self.preload()

    def preload(self):
        """
        Preloads data into memory by iterating through directories and files in the specified root directory.
        Returns:
            None
        """
        total_files = sum(len(files) for _, _, files in os.walk(self.root))
        with tqdm(
            total=total_files, desc="[spgdataset/MEMCACHE] Preloading files into memory"
        ) as pbar:
            for root, dirs, files in os.walk(self.root):
                for file in files:
                    pbar.update(1)
                    # check that file fits mask
                    if fnmatch(file, self.mask):
                        path = os.path.join(root, file)
                        with open(path, "rb") as f:
                            if self.mode == "pickle":
                                data = pickle.load(f)
                            else:
                                data = load(f)

                        id = path.replace(self.root, "").strip("/")
                        res = self.add(id, data)
                        if res == Status.FULL:
                            return None

    def get(self, path: str) -> Tensor | None:
        """
        Retrieve a Tensor from the storage based on the given path.
        Args:
            path (str): The file path used to identify the Tensor in the storage.
        Returns:
            Tensor | None: The Tensor object if found in the storage, otherwise None.
        """
        id = path.replace(self.root, "").strip("/")
        return self.__get(id)

    def __get(self, id: str) -> Tensor | None:
        """
        Retrieves a tensor from the storage by its identifier (relative path without root part).
        Args:
            id (str): The identifier of the tensor to retrieve.
        Returns:
            Tensor | None: The tensor if found in storage, otherwise None.
        """
        if id in self.storage.keys():
            return self.storage[id]
        else:
            return None

    def add(self, id: str, obj: np.ndarray | Tensor) -> int:
        """
        Adds an object to the memory storage.
        Args:
            id (str): The identifier (path without root) for the object to be added.
            obj (np.ndarray | Tensor): The object to be added, either as a NumPy array or a Tensor.
        Returns:
            int: Status code indicating the result of the operation.
                - Status.FULL: If the storage is full.
                - Status.ALREADY_EXISTS: If the object with the given id already exists in the storage.
                - Status.OK: If the object was successfully added to the storage.
        """
        if self.full is True:
            return Status.FULL
        elif id in self.storage.keys():
            return Status.ALREADY_EXISTS
        else:
            if isinstance(obj, np.ndarray):
                obj = tensor(obj, dtype=float32)
            required_size = sys.getsizeof(id) + self.get_object_size(obj)
            if self.get_space_available() < required_size:
                self.full = True
                logging.warning(
                    "[spgdataset/MEMCACHE] [!] Memory limit reached. You may try to increase the limit."
                )
                return Status.FULL
            self.storage[id] = obj
            self.running_size += required_size
            return Status.OK

    @staticmethod
    def get_object_size(obj: np.ndarray | Tensor) -> int:
        """
        Calculates the size of a given object in bytes.
        Args:
            obj (np.ndarray | Tensor): The object whose size is to be calculated.
                                    It can be either a NumPy ndarray or a PyTorch Tensor.
        Returns:
            int: The size of the object in bytes.
        Raises:
            TypeError: If the object is not a NumPy ndarray or a PyTorch Tensor.
        """

        if isinstance(obj, Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            raise TypeError("Unsupported type: must be np.ndarray or torch.Tensor")

    def get_space_available(self):
        """
        Calculates the available remaining space.
            Returns:
                int: The available space in bytes.
        """
        return self.limit * 1048576 - self.running_size

    def __len__(self):
        return len(self.storage)
