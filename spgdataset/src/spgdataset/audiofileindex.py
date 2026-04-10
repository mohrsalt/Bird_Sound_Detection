import os
import logging
import json
import pickle
from . import audio as _audio
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np


####################################################################
#                 AudioFileIndex/AudioFileRecord
####################################################################
# - AudioFileIndex is a class that represents an index of audio files
# in a directory. It is used to store metadata and other information
# about the audio files in a structured way.
# - AudioFileRecord is a class that represents a single audio
# file in the index.


# fmt: off
@dataclass(slots=True)
class AudioFileRecord:
    path: str          # Path to raw audio file without audio_root
    length: int        # Length of the raw audio file
    intervals: dict    # dictionary containing various intervals, could be just SPEECH or MALE, FEMALE, etc.
    slices: np.ndarray # Slices of the audio file with dtype=[('int_field', np.int32), ('float_field', np.float16)]
    metadata: dict     # Metadata associated with the audio file
# fmt: on


class AudioFileIndex:
    def __init__(
        self,
        root_dir: str,
        metadata_json_path: str,
        intervals_names: list = [],
        metadata_fields: list = [],
        extensions: list = _audio.AUDIO_EXT,
        sample_rate: int = 32000,
    ):
        """
        Initialize the AudioFileIndex.
        Args:
            root_dir (str): The root directory where audio files are stored.
            metadata_json_path (str): Path to the JSON file containing metadata.
            intervals_names (list, optional): List of interval names to be loaded from metadata and used. Defaults to an empty list.
            metadata_fields (list, optional): List of metadata fields to be loaded from metadata and used. Defaults to an empty list.
            extensions (list, optional): List of audio file extensions to be considered. Defaults to _audio.AUDIO_EXT.
            sample_rate (int, optional): The sample rate for audio files. Defaults to 16000.
        """
        assert sample_rate == 32000, "Only 16kHz sample rate is supported"
        self.config = SimpleNamespace(
            root_dir=root_dir, extensions=extensions, sample_rate=sample_rate
        )
        self.mapping = {}
        self.index = self._scan_directory(root_dir)
        self._update_with_metadata(
            metadata_json_path=metadata_json_path,
            intervals_names=intervals_names,
            metadata_fields=metadata_fields,
        )

    def _scan_directory(self, root_dir: str) -> dict:
        """
        Recursively scans the given root directory and builds a tree structure of audio files.
        If the root directory is not set in the configuration, it sets it to the provided root_dir.
        Otherwise, it asserts that the provided root_dir is a subdirectory of the configured root directory.
        Args:
            root_dir (str): The root directory to scan for audio files.
        Returns:
            dict: A nested dictionary representing the directory tree structure, where keys are directory
                  or file names and values are either further nested dictionaries (for directories) or
                  AudioFileRecord instances (for audio files).
        Raises:
            AssertionError: If the provided root_dir is not a subdirectory of the configured root directory.
        Logs:
            Warning: If a file with an unsupported extension is found.
        """
        if self.config.root_dir is None:
            self.config.root_dir = root_dir
        else:
            assert self.config.root_dir in root_dir, "Root directory mismatch"
        tree = {}
        with os.scandir(root_dir) as it:
            for entry in it:
                if entry.is_dir():
                    tree[entry.name] = self._scan_directory(entry.path)
                else:
                    if entry.name.endswith(tuple(self.config.extensions)):
                        self.mapping[len(self.mapping)] = (
                            entry.path.replace(self.config.root_dir, "")
                            .strip("/")
                            .split("/")
                        )
                        tree[entry.name] = AudioFileRecord(
                            path=entry.path.replace(self.config.root_dir, "").strip(
                                "/"
                            ),
                            length=-1,
                            intervals={},
                            slices=np.array(
                                [],
                                dtype=[
                                    ("int_field", np.int32),
                                    ("float_field", np.float16),
                                ],
                            ),
                            metadata={},
                        )
                    else:
                        logging.warning(
                            f"[spgdataset/AudioFileIndex] Found file with unsupported extension: {entry.name}"
                        )
        return tree

    def _update_with_metadata(
        self, metadata_json_path: str, intervals_names: list, metadata_fields: list
    ):
        """
        Updates the file index with metadata from a JSON file.
        Args:
            metadata_json_path (str): Path to the metadata JSON file.
            intervals_names (list): List of interval names to be read from metadata and updated.
            metadata_fields (list): List of metadata fields to be read from metadata and updated.
        Raises:
            AssertionError: If the metadata file is not in JSON format or does not exist.
            IOError: If there is an error loading the metadata file.
            Exception: If a file in the index is missing in the metadata, or if required fields are missing in the metadata.
        Returns:
            self.index: The updated file index.
        """
        assert metadata_json_path.endswith(
            ".json"
        ), "Only json format is supported for metadata"
        assert os.path.exists(
            metadata_json_path
        ), f"Metadata file {metadata_json_path} does not exist"
        try:
            with open(metadata_json_path, "rt") as f:
                metadict = json.load(f)
        except Exception as e:
            raise IOError(f"Unable to load metadata file: {str(e)}")
        logging.info("[spgdataset/AudioFileIndex] Updating file index with metadata")

        for _, record in self:
            # record itself
            if record.path in metadict:
                metadata = metadict[record.path]
            else:
                raise Exception(
                    f"[spgdataset/AudioFileIndex] File {record.path} is missing in metadata"
                )
            # length
            if "length" not in metadata:
                logging.warning(
                    f"[spgdataset/AudioFileIndex] File {record.path} is missing field `length` in metadata, reading from file"
                )
                record.length = _audio.audio_load_length(
                    os.path.join(self.config.root_dir, record.path),
                    sr=self.config.sample_rate,
                )
            else:
                record.length = metadata["length"]
            # intervals
            if len(intervals_names) > 0 and "intervals" not in metadata:
                raise Exception(
                    f"[spgdataset/AudioFileIndex] File {record.path} is missing field `intervals` in metadata"
                )
            for interval_name in intervals_names:
                if interval_name not in metadata["intervals"]:
                    raise Exception(
                        f"[spgdataset/AudioFileIndex] File {record.path} is missing field `{interval_name}` in intervals"
                    )
                record.intervals[interval_name] = np.array(
                    metadata["intervals"][interval_name], dtype=np.int32
                )
            # metadata
            if len(metadata_fields) > 0 and "metadata" not in metadata:
                raise Exception(
                    f"[spgdataset/AudioFileIndex] File {record.path} is missing field `metadata` in metadata"
                )
            for field_name in metadata_fields:
                if field_name not in metadata["metadata"]:
                    raise Exception(
                        f"[spgdataset/AudioFileIndex] File {record.path} is missing field `{field_name}` in metadata"
                    )
                record.metadata[field_name] = metadata["metadata"][field_name]

        return self.index

    # region get/set methods
    def get_by_index(self, index: int) -> AudioFileRecord:
        """
        Retrieve an AudioFileRecord object by the given index.
        Args:
            index (int): The index of the AudioFileRecord object in the mapping.
        Returns:
            AudioFileRecord: The AudioFileRecord object if found, otherwise None.
        """
        if index in self.mapping:
            trail = self.mapping[index]
            return self.get_by_trail(trail)
        return

    def get_by_trail(self, trail: list) -> AudioFileRecord:
        """
        Retrieve an AudioFileRecord object by the given trail.
        Args:
            trail (list): The trail of keys to traverse the index tree.
        Returns:
            AudioFileRecord: The AudioFileRecord object if found, otherwise None.
        """
        current_tree = self.index
        for part in trail:
            if part in current_tree:
                current_tree = current_tree[part]
            else:
                return None
        return current_tree

    def set_by_index(self, index: int, record: AudioFileRecord):
        """
        Set an AudioFileRecord object in the index tree by the given index.
        Args:
            index (int): The index of the AudioFileRecord object in the mapping.
            record (AudioFileRecord): The AudioFileRecord object to set.
        """
        if index in self.mapping:
            trail = self.mapping[index]
            self.set_by_trail(trail, record)

    def set_by_trail(self, trail: list, record: AudioFileRecord):
        """
        Set an AudioFileRecord object in the index tree by the given trail.
        Args:
            trail (list): The trail of keys to traverse the index tree.
            record (AudioFileRecord): The AudioFileRecord object to set.
        """
        current_tree = self.index
        for i, part in enumerate(trail):
            if i == len(trail) - 1:
                current_tree[part] = record
            else:
                if part not in current_tree:
                    current_tree[part] = {}
                current_tree = current_tree[part]

    # endregion

    # region save/load methods
    def save(self, path: str):
        """
        Save the current state of itself to files.
        This method saves the `mapping`, `config`, and `index` attributes of the object
        to separate files with the specified path as the base name. The files will have
        the extensions `.mapping`, `.config`, and `.index` respectively.
        Args:
            path (str): The base path where the files will be saved. The method will
                        create the necessary directories if they do not exist.
        Raises:
            OSError: If there is an error creating directories or writing to the files.
        """

        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(path + ".mapping", "wb") as f:
            pickle.dump(self.mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path + ".config", "wb") as f:
            pickle.dump(self.config, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path + ".index", "wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

    # check if all files exist
    @staticmethod
    def files_exist(path: str):
        if not os.path.exists(os.path.dirname(path)):
            return False
        extensions = [".mapping", ".index", ".config"]
        for ext in extensions:
            if not os.path.exists(path + ext):
                return False
        return True

    def load(self, path: str):
        """
        Load the mapping, config, and index data from the specified path.
        Args:
            path (str): The base path to load the files from. The method expects
                        three files with extensions .mapping, .config, and .index
                        to be present at this path.
        Raises:
            AssertionError: If the files do not exist at the specified path.
        Side Effects:
            Sets the instance variables `mapping`, `config`, and `index` with the
            data loaded from the respective files.
        """

        assert self.files_exist(path), f"Can't load from {path}"
        with open(path + ".mapping", "rb") as f:
            self.mapping = pickle.load(f)
        with open(path + ".config", "rb") as f:
            self.config = pickle.load(f)
        with open(path + ".index", "rb") as f:
            self.index = pickle.load(f)

    # endregion

    def __iter__(self):
        for index in self.mapping:
            yield (index, self.get_by_index(index))
