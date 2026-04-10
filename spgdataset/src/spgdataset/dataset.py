import numpy as np
import torch
from spgdataset.src.spgdataset import audio as _audio
from spgdataset.src.spgdataset.utils import exact_div, exact_mult
from spgdataset.src.spgdataset.memcache import MemoryStorage
from spgdataset.src.spgdataset.audiofileindex import AudioFileIndex
import os
import logging
import pickle
from math import floor, ceil
from types import SimpleNamespace

####################################################################
#                     Spectrogram Dataset
####################################################################


class SpectrogramDataset:
    def __init__(
        self,
        audio_root: str,
        spectrograms_root: str,
        index_root: str,
        metadata_json_path: str | None = None,
        sample_rate: int = 32000,
        window_size_sec: float = 1.92,
        window_offset_sec: float = 0.1,
        window_content_ratio: float = 0.5,
        hop_length: int = 320,
        n_mels: int = 128,
        n_fft: int = 800,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        split: tuple = (1,),
        # fmt: off
        output_configuration: dict | None = None, 
        max_memory_cache: int = 0,  # max memory space to retain spectrograms
        num_workers: int = 0,
        # number of workers for multiprocessing tasks
        # fmt: on
    ) -> None:
        """
        Initialize the dataset with the given parameters.
        Args:
            audio_root (str): ..........................Path to the root directory containing audio files.
            spectrograms_root (str): ...................Path to the root directory containing (or to store) spectrogram files.
            index_root (str): ..........................Path to the directory containing (or to store) index files.
            metadata_json_path (str): ..................Path to the metadata JSON file.
            sample_rate (int, optional): ...............Sample rate for audio processing. Defaults to 16000.
            window_size_sec (float, optional): .........Window size in seconds for spectrogram generation. Defaults to 1.92.
            window_offset_sec (float, optional): .......Window offset in seconds for spectrogram generation. Defaults to 0.1.
            window_content_ratio (float, optional): ....Ratio of content within the window. Defaults to 0.5.
            hop_length (int, optional): ................Hop length for spectrogram generation. Defaults to 160.
            n_mels (int, optional): ....................Number of Mel bands to generate. Defaults to 80.
            n_fft (int, optional): .....................Number of FFT components. Defaults to 400.
            normalize (bool, optional): ................Whether to normalize the spectrograms. Defaults to True.
            dtype (torch.dtype, optional): .............Data type for tensors. Defaults to torch.float32.
            split (tuple, optional): ...................Split ratio for the dataset (train/validate/test). Defaults to (1,).
            output_configuration (dict, optional): .....Configuration for the output data. Defaults to a predefined dictionary.
            max_memory_cache (int, optional): ..........Maximum memory space to retain spectrograms. Defaults to 0.
            num_workers (int, optional): ...............Number of workers for multiprocessing tasks. Defaults to 0.
        Returns:
            None
        """
        spg_fps = exact_div(sample_rate, hop_length)
        self.config = SimpleNamespace(
            audio_root=audio_root,
            spg_root=spectrograms_root,
            index_root=index_root,
            sample_rate=sample_rate,
            window_size_sec=window_size_sec,
            window_offset_sec=window_offset_sec,
            window_size=exact_mult(window_size_sec, sample_rate),
            window_offset=exact_mult(window_offset_sec, sample_rate),
            content_ratio=window_content_ratio,
            spg_fps=spg_fps,
            spg_length=exact_mult(window_size_sec, spg_fps),
            spg_offset=exact_mult(window_offset_sec, spg_fps),
            hop_length=hop_length,
            n_mels=n_mels,
            n_fft=n_fft,
            normalize=normalize,
            metadata_json_path=metadata_json_path,
            max_memory_cache=max_memory_cache,
            num_workers=num_workers,
        )
        self.dtype = dtype
        self.__split_ratio = (1,)
        self.__mode = 0
        self.fraction = 1.0
        self.__fraction_filter = None

        self._configure_output(output_configuration)
        # load existing or build self.file_index
        if self._load_file_index() is False:
            self._build_file_index()
            self._save_file_index()

        self.split(split)
        # internally followed by:
        #   self._update_retrieval_index()

        self._mass_generate_spectrograms()

        if self.__output_configuration["label"] is not None:
            self._build_labels(label_name=self.__output_configuration["label"])
        else:
            self.classes = []

        self._update_slices()
        self._update_retrieval_index()
        self.validate_dataset()

        self.memcache = MemoryStorage(
            root=spectrograms_root,
            mask="*" + self._generate_cache_filename(""),
            limit=max_memory_cache,
        )

    def _configure_output(self, configuration: dict):
        # fmt: off
        default_configuration: dict = {
                "audio": False,       # return raw audio
                "spectrogram": True,  # return spectrogram
                "masks": [],          # return masks ['speech', 'male_speech'] etc, must present in metadata.json 'intervals'
                "meta": [],           # return additional properties: ['gender', 'speaker'] etc, must present in metadata.json 'metadata'
                "label": str
                | None,               # property from 'meta' to use as a label, if None, no labels will be returned
            }
        # set up configuration field types
        ftypes = {
            "audio": bool,
            "spectrogram": bool,
            "masks": list,
            "meta": list,
            "label": str | None
        }
        # validate configuration values
        if configuration is None:
            raise ValueError("Output configuration is not provided")
        if not isinstance(configuration, dict):
            raise ValueError("Output configuration should be a dictionary")
        for k, v in configuration.items():
            assert k in default_configuration, f"Unknown property {k} in output configuration"
            assert isinstance(v, ftypes[k]), f"Invalid type {type(v)} for {k} in output configuration, expected {ftypes[k]}"



        if "_SpectrogramDataset__output_configuration" not in self.__dict__:  
            self.__output_configuration = {}
        # fmt: on
        for k, v in configuration.items():
            assert (
                k in default_configuration
            ), f"Unknown property {k} in output configuration"
            self.__output_configuration[k] = v
        for k, v in default_configuration.items():
            if k not in self.__output_configuration:
                self.__output_configuration[k] = v

    # region index property (AudioFileIndex)
    def _load_file_index(self) -> bool:
        """
        Loads the file index for the dataset if it exists.

        Returns:
            bool: True if the index file exists and is loaded successfully, False otherwise.
        """
        index_name = self._generate_index_name()
        index_path = os.path.join(self.config.index_root, index_name)
        if AudioFileIndex.files_exist(index_path) is False:
            return False
        else:
            logging.info(
                f"[spgdataset] Loading index for dataset '{self.config.audio_root}'"
            )
            self.file_index = AudioFileIndex(
                root_dir=self.config.audio_root,
                metadata_json_path=self.config.metadata_json_path,
                sample_rate=self.config.sample_rate,
            )
            self.file_index.load(index_path)
            return True

    def _save_file_index(self) -> None:
        """
        Saves the file index for the dataset.
        """
        index_name = self._generate_index_name()
        index_path = os.path.join(self.config.index_root, index_name)
        self.file_index.save(index_path)

    def _build_file_index(self) -> None:
        """
        Scans audio dir, builds index, then saves a copy to the disk.
        """
        logging.info(
            f"[spgdataset] Building index for dataset '{self.config.audio_root}'"
        )
        self.file_index = AudioFileIndex(
            root_dir=self.config.audio_root,
            metadata_json_path=self.config.metadata_json_path,
            intervals_names=self.__output_configuration["masks"],
            metadata_fields=self.__output_configuration["meta"],
            sample_rate=self.config.sample_rate,
        )

    def _generate_index_name(self):
        index_substr = ("A" if self.__output_configuration["audio"] else "") + (
            "S" if self.__output_configuration["spectrogram"] else ""
        )
        if len(self.__output_configuration["masks"]) > 0:
            index_substr += (
                "M("
                + "_".join([mask for mask in self.__output_configuration["masks"]])
                + ")"
            )
        index_substr += "L" if self.__output_configuration["label"] is not None else ""
        if len(self.__output_configuration["meta"]) > 0:
            index_substr += (
                "X("
                + "_".join([meta for meta in self.__output_configuration["meta"]])
                + ")"
            )
        index_name = f'dataset_{self.config.audio_root.split("/")[-1]}_{self.config.n_mels}x{self.config.spg_length}_{index_substr}'
        return index_name

    # endregion

    def _build_labels(self, label_name: str):
        """
        Builds labels for the dataset.

        This method iterates through the file index and extracts the specified label
        from the metadata of each record. It raises an exception if the label is
        missing in any record. The unique labels are then sorted and stored in the
        `self.classes` attribute.

        Args:
            label_name (str): The name of the label to extract from the metadata.

        Raises:
            Exception: If the specified label is missing in any record's metadata.
        """
        logging.info("[spgdataset] Building labels for dataset")
        labels_all = []
        for _, record in self.file_index:
            if label_name not in record.metadata:
                raise Exception(
                    f"{label_name} is missing in records for file {record.path}"
                )
            labels_all.append(record.metadata[label_name])
        self.classes = sorted(list(set(labels_all)))

    def _mass_generate_spectrograms(self):
        """
        Generates spectrograms
        """

        def generate_spectrogram_filename(audio_filename: str):
            return f"{audio_filename.split('.')[0]}_{self.config.window_size_sec:.2f}s_{self.config.n_mels}x{self.config.spg_length}.sc"

        logging.info("[spgdataset] Looking for files without spectrograms")
        unprocessed = []
        for _, record in self.file_index:
            filename = os.path.basename(record.path)
            filepath = os.path.join(self.config.audio_root, record.path)
            spectrogram_filename = generate_spectrogram_filename(filename)
            spectrogram_filepath = filepath.replace(self.config.audio_root, self.config.spg_root).replace(filename, spectrogram_filename)  # fmt: skip
            if os.path.exists(spectrogram_filepath):
                continue
            else:
                unprocessed.append(
                    {"file": filepath, "cache_file": spectrogram_filepath}
                )

        if len(unprocessed) > 0:
            logging.info(
                f"[spgdataset] {len(unprocessed)} files without spectrograms found, generating.."
            )
            AP = _audio.MassProcessor(
                handler_fn=_audio.spectrogram_generator,
                num_workers=self.config.num_workers,
                progress_string=" + Generating spectrograms",
                settings={
                    "root": self.config.spg_root,
                    "sr": self.config.sample_rate,
                    "n_mels": self.config.n_mels,
                    "n_fft": self.config.n_fft,
                    "hop_length": self.config.hop_length,
                    "wsize": self.config.window_size,
                    "woffset": self.config.window_offset,
                    "normalize": self.config.normalize,
                    "dtype": self.dtype,
                },
            )
            AP(unprocessed)

    # region slices
    def _update_slices(self):
        """
        Computes slices for each audio file, based on speech intervals and content ratio.
        """
        for _, record in self.file_index:
            slices = []
            windows = self._calculate_sliding_windows(
                record.length, self.config.window_size, self.config.window_offset
            )

            # if no masks, consider all windows with content ratio = 1
            if len(self.__output_configuration["masks"]) == 0:
                slices = [(offset, 1) for offset in range(len(windows))]
            else:
                # merge intervals (numpy arrays)
                content_intervals = np.concatenate(
                    [ival for ival in record.intervals.values()], axis=0
                )

                # Merge overlapping intervals
                if len(content_intervals) > 0:
                    content_intervals = content_intervals[
                        np.argsort(content_intervals[:, 0])
                    ]
                    merged_intervals = [content_intervals[0].tolist()]

                    for start, end in content_intervals[1:]:
                        last_end = merged_intervals[-1][1]
                        if start <= last_end:
                            merged_intervals[-1][1] = max(last_end, end)
                        else:
                            merged_intervals.append([start, end])

                    content_intervals = np.array(merged_intervals)
                else:
                    content_intervals = np.array([], dtype=np.int32).reshape(0, 2)

                overlaps = self._calculate_overlaps(windows, content_intervals)
                for offset, overlap in enumerate(overlaps):
                    if overlap >= self.config.content_ratio:
                        slices.append((offset, overlap.item()))
            record.slices = np.array(
                slices, dtype=[("int_field", np.int32), ("float_field", np.float16)]
            )

    @staticmethod
    def _calculate_sliding_windows(total_length: int, window_size: int, offset: int):
        """
        Calculates sliding windows for a given total length, window size, and offset.

        Args:
            total_length (int): The total length of the sequence.
            window_size (int): The size of each sliding window.
            offset (int): The offset between the start of each window.
        Returns:
            np.ndarray: A 2D array where each row represents a window with start and end positions.
                    The array has shape (num_windows, 2) and dtype np.uint32.
        """

        if total_length <= window_size:
            return np.array([[0, window_size]], dtype=np.uint32)
        num_windows = int((total_length - window_size) / offset) + 1
        start_positions = np.arange(0, num_windows * offset, offset, dtype=np.uint32)[
            :num_windows
        ]
        end_positions = start_positions + window_size
        windows = np.column_stack((start_positions, end_positions))
        return windows

    @staticmethod
    def _calculate_overlaps(
        sliding_windows: np.ndarray, intervals: np.ndarray
    ) -> np.ndarray:
        """
        Computes the overlap between sliding windows and intervals in percent to window size.

        Args:
            sliding_windows: np.array of shape (num_windows, 2) where each row is [start, stop]
            intervals: np.array of shape (num_intervals, 2) where each row is [start, stop]

        Returns:
            2D numpy array of shape (num_windows, num_intervals) containing the overlap percentages.
        """
        # Extract start and stop points
        window_starts = sliding_windows[:, 0][:, np.newaxis]  # Shape (num_windows, 1)
        window_stops = sliding_windows[:, 1][:, np.newaxis]  # Shape (num_windows, 1)

        interval_starts = intervals[:, 0]  # Shape (num_intervals,)
        interval_stops = intervals[:, 1]  # Shape (num_intervals,)

        # Calculate overlap start and end points
        overlap_starts = np.maximum(
            window_starts, interval_starts
        )  # Shape (num_windows, num_intervals)
        overlap_stops = np.minimum(
            window_stops, interval_stops
        )  # Shape (num_windows, num_intervals)

        # Calculate overlap lengths
        overlap_lengths = np.maximum(
            0, overlap_stops.astype(np.int64) - overlap_starts.astype(np.int64)
        )  # Shape (num_windows, num_intervals)

        # Assuming all windows have the same size
        window_size = window_stops[0] - window_starts[0]  # Size of the first window

        # Calculate overlaps with respect to the window size
        overlaps = (
            np.sum(overlap_lengths, axis=1) / window_size
        )  # Shape (num_windows, num_intervals)

        return overlaps

    # endregion

    def _update_retrieval_index(self) -> list:
        self.retrieval_index = [[] for _ in range(len(self.__split_ratio))]

        def sort_pool(pool):
            if len(self.__split_ratio) == 1:
                # aka no splits
                self.retrieval_index[0] += pool
            else:
                # randomly distribute elements between splits according to probs
                flips = np.random.uniform(0, 1, size=len(pool))
                cumsums = [0] + list(np.cumsum(self.__split_ratio))
                distrib = [
                    np.where((flips >= cumsums[i]) & (flips < cumsums[i + 1]))[0]
                    for i in range(len(cumsums) - 1)
                ]
                assert sum([len(arr) for arr in distrib]) == len(pool)
                for mode, arr in enumerate(distrib):
                    self.retrieval_index[mode] += [pool[ix] for ix in arr]

        pool = []
        for i, record in self.file_index:
            for j in range(0, len(record.slices)):
                pool.append((i, j))
                if len(pool) % 1000 == 0:
                    sort_pool(pool)
                    pool = []
        sort_pool(pool)
        return self.retrieval_index

    def validate_dataset(self) -> bool:
        """
        Validates the file index for required fields to be present.
        This method checks if the necessary fields are present in the file index and raises exceptions or warnings if any required fields are missing or invalid.

        Returns:
            bool: True if the dataset is valid, False otherwise.
        """
        logging.info("[spgdataset] Validating dataset")
        for _, record in self.file_index:
            if len(record.slices) == 0:
                logging.warning(
                    f"File {record.path} contains no slices, consider removing it from the dataset"
                )
            if record.length is None:
                raise Exception(f"File {record.path} is missing field 'length'")
            if record.intervals is None:
                raise Exception(f"File {record.path} is missing field 'intervals'")
            if len(record.intervals) == 0:
                logging.warning(f"File {record.path} contains no intervals")
            for mask_name in self.__output_configuration["masks"]:
                if mask_name not in record.intervals:
                    logging.warning(
                        f"File {record.path} contains no intervals for '{mask_name}'"
                    )
            for slice in record.slices:
                assert (
                    slice[1] + 1e-4
                ) >= self.config.content_ratio, (
                    "Validation failed: slice with insufficient amount of content"
                )
            for metadata_field in self.__output_configuration["meta"]:
                if metadata_field not in record.metadata:
                    logging.warning(
                        f"File {record.path} contains no metadata field '{mask_name}'"
                    )
            if self.__output_configuration["label"] is True:
                assert (
                    self.classes is not None and len(self.classes) > 0
                ), "Validation failed: no classes found while class labels were provided"
        return True

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Value depends on the current mode and fraction:
        - if dataset is split between different modes, returns the number of samples for the current mode
        - if fraction is set, returns the number of samples based on the fraction value

            For example, if dataset is split (0.8, 0.2), current mode is 0 (train), and fraction is 0.5,
            then the number of samples will be 50% of the samples selected for mode 0 (training)
        """
        if self.__fraction_filter is not None:
            return len(self.__fraction_filter)
        else:
            return len(self.retrieval_index[self.__mode])

    def get_output_configuration(self):
        """Returns current output configuration"""
        return self.__output_configuration

    def __getitem__(self, i: int):
        """
        Retrieves a data sample from the dataset at the given index.
        Args:
            i (int): Index of the data sample to retrieve.
        Returns:
            dict: A dictionary containing the requested data components based on the output configuration.
                Possible keys in the dictionary include:
                - "audio": The audio data if audio output is enabled.
                - "spectrogram": The spectrogram data if spectrogram output is enabled.
                - "label": The label of the data sample if label output is enabled.
                - "masks": A dictionary of masks if mask output is enabled.
                - "meta": A dictionary of metadata fields if meta output is enabled.
        """

        if self.__fraction_filter is not None:
            index = self.__fraction_filter[i]
        else:
            index = i
        record_id, slice_num = self.retrieval_index[self.__mode][index]
        record = self.file_index.get_by_index(record_id)

        output = {}
        # audio
        if self.__output_configuration["audio"] is True:
            filename = os.path.join(
                self.config.audio_root,
                record.path,
            )
            offset = self.config.window_offset * slice_num
            audio = _audio.load_audio(filename=filename, sr=self.config.sample_rate)[
                offset : offset + self.config.window_size
            ]
            output["audio"] = audio
        # spectrogram
        if self.__output_configuration["spectrogram"] is True:
            spectrogram = self._get_spectrogram(record_id, slice_num)
            output["spectrogram"] = spectrogram
        # label
        if self.__output_configuration["label"] is not None:
            label = self.classes.index(
                record.metadata[self.__output_configuration["label"]]
            )
            output["label"] = label
        # masks
        if len(self.__output_configuration["masks"]) > 0:
            output["masks"] = {}
            for mask_name in self.__output_configuration["masks"]:
                mask = self._compute_mask(record_id, slice_num, mask_name=mask_name)
                output["masks"][mask_name] = mask
        # meta
        if (
            len(self.__output_configuration["meta"]) == 1
            and self.__output_configuration["label"] is not None
            and self.__output_configuration["label"]
            == self.__output_configuration["meta"]
        ):
            pass
        elif len(self.__output_configuration["meta"]) > 0:
            output["meta"] = {}
            for metadata_field in self.__output_configuration["meta"]:
                if (
                    self.__output_configuration["label"] is not None
                    and metadata_field == self.__output_configuration["label"]
                ):
                    continue
                output["meta"][metadata_field] = record.metadata[metadata_field]
        return output

    # region spectrogram
    def _get_spectrogram(self, record_id: int, slice_num: int) -> torch.Tensor:
        """
        Retrieves a spectrogram for a given record and slice number.
        This method attempts to retrieve the spectrogram from memory cache first.
        If it is not found in the cache, it reads the spectrogram from the file system.
        The spectrogram is then sliced according to the provided slice number.
        Args:
            record_id (int): The ID of the record to retrieve the spectrogram for.
            slice_num (int): The slice number to extract from the spectrogram.
        Returns:
            torch.Tensor: The spectrogram slice as a tensor.
        Raises:
            AssertionError: If the shape of the retrieved spectrogram does not match the expected shape.
        """
        record = self.file_index.get_by_index(record_id)

        cache_path = os.path.join(
            self.config.spg_root, self._generate_cache_filename(record.path)
        )
        # try to read from memory cache
        large_spectrogram = self.memcache.get(cache_path)
        if large_spectrogram is None:
            large_spectrogram = self._spectrogram_file_read(cache_path)

        slice = record.slices[slice_num]
        spg_start = self.config.spg_offset * slice[0]
        spectrogram = large_spectrogram[
            :, spg_start : spg_start + self.config.spg_length
        ]
        expected_shape = torch.Size([self.config.n_mels, self.config.spg_length])
        assert (
            spectrogram.shape == expected_shape
        ), f"Spectrogram shape incorrect: expected {expected_shape}, got {spectrogram.shape}"
        return spectrogram  # type: ignore

    def _spectrogram_file_read(self, path: str) -> torch.Tensor:
        """
        Reads a spectrogram from a file
        Args:
            path (str): The path to the spectrogram file.
        Returns:
            torch.Tensor: The spectrogram loaded from the file.
        Raises:
            IOError: If the file does not exist at the given path.
        """
        if not os.path.isfile(path):
            raise IOError(f"Cache file (spectrogram) does not exist at {path}")
        with open(path, "rb") as f:
            spectrogram = pickle.load(f)
        return spectrogram

    # endregion

    # region speech/silence mask
    def _compute_mask(
        self, record_id: int, slice_num: int, mask_name: str
    ) -> torch.Tensor:
        """
        Computes a mask tensor for a given record and slice number based on specified intervals.
        Args:
            record_id (int): The ID of the record to process.
            slice_num (int): The slice number within the record.
            mask_name (str): The name of the mask (same as intervals[name]) to compute.
        Returns:
            torch.Tensor: A tensor representing the computed mask.
        Raises:
            ValueError: If global speech intervals are not provided for the file.
            AssertionError: If the computed mask does not meet the content ratio requirements.
        Notes:
            - The mask is computed based on the intersection of audio intervals and provided intervals.
            - The mask tensor is initialized to zeros and updated based on the intersected intervals.
            - The function ensures that the computed mask meets the specified content ratio.
        """

        def rf(x: int):
            return floor(x / self.config.hop_length)

        def rc(x: int):
            return ceil(x / self.config.hop_length)

        record = self.file_index.get_by_index(record_id)
        intervals = record.intervals[mask_name]
        if intervals is None:
            raise ValueError("Global speech intervals are not provided for file")

        slice = record.slices[slice_num]
        audio_start = self.config.window_offset * slice[0]
        audio_interval = [audio_start, audio_start + self.config.window_size]
        mask = torch.zeros(self.config.spg_length, dtype=self.dtype)
        ii = self._intersect_interval_arrays([audio_interval], intervals)
        iispg = [[rf(iv[0]), min(rc(iv[1]), rc(record.length) - 1)] for iv in ii]
        start = rf(audio_start)
        for iv in iispg:
            mask[iv[0] - start : iv[1] - start] = 1.0
        if len(self.__output_configuration["masks"]) == 1:
            assert self.config.content_ratio <= sum(mask) <= len(mask), "Incorrect mask"
        return mask

    @staticmethod
    def _intersect_interval_arrays(arr1: list, arr2: list | np.ndarray):
        """
        Find the intersections between two arrays of intervals.
        This function takes two arrays of intervals and returns a list of intervals
        that represent the intersections between the intervals in the two arrays.
        Args:
            arr1 (list): A list of intervals, where each interval is represented as a list or tuple of two elements [start, end].
            arr2 (list | np.ndarray): A list or numpy array of intervals, where each interval is represented as a list or tuple of two elements [start, end].
        Returns:
            list: A list of intervals representing the intersections between the intervals in arr1 and arr2. Each interval is represented as a list of two elements [start, end].
        """
        if isinstance(arr2, np.ndarray):
            arr2 = arr2.tolist()
        if len(arr2) == 0:
            return []

        i, j = 0, 0
        intersections = []
        # Loop through both arrays of intervals
        while i < len(arr1) and j < len(arr2):
            start1, end1 = arr1[i]
            start2, end2 = arr2[j]

            # Find the intersection between the two intervals
            intersection_start = max(start1, start2)
            intersection_end = min(end1, end2)

            # If the intersection is valid, add it to the result
            if intersection_start < intersection_end:
                intersections.append([intersection_start, intersection_end])

            # Move to the next interval in the array that finishes earlier
            if end1 < end2:
                i += 1
            else:
                j += 1

        return intersections

    # endregion

    def _generate_cache_filename(self, filename: str) -> str:
        """
        Generates a cache filename based on the provided filename and configuration parameters.
        Args:
            filename (str): The original filename to be transformed.
        Returns:
            str: The generated cache filename, which includes the original filename (without extension),
                 the window size in seconds, the number of mel bands, and the spectrogram length.
        """
        ftrim = ".".join(filename.split(".")[:-1])
        return f"{ftrim}_{float(self.config.window_size_sec):.2f}s_{self.config.n_mels}x{self.config.spg_length}.sc"

    # region split property
    @property
    def split_ratio(self):
        """
        Split ratio property
        Splits dataset into parts, if you want to divide it into train, validate and test parts (in any combination)
        """
        return self.__split_ratio

    @split_ratio.setter
    def split_ratio(self, ratio: tuple):
        assert sum(ratio) == 1, "Split parts should sum to 1"
        self.__split_ratio = ratio
        self._update_retrieval_index()

    def split(self, ratio: tuple):
        # run @setter
        self.split_ratio = ratio

    # endregion

    # region fraction property

    ##################
    # Fraction property to use only a fraction of samples for the selected mode
    ##################

    @property
    def fraction(self):
        return self.__fraction

    @fraction.setter
    def fraction(self, value: float = 1.0):
        """
        If you want to speed things up, randomly chooses fraction amount of samples to be used
        out of selected for the current mode (training, validation etc).

        Args:
            value (float): The fraction of samples to be used. If set to 0.5, half of the samples
                                  will be used for training and half for validation. If 1, no filtering
                                  is applied.
        Attributes:
            __fraction (float): Stores the input fraction value.
            fraction_filter (numpy.ndarray | None): An array of indices indicating which samples belong
                                                    to the current mode based on the fraction value.

        Examples:
            - If fraction is 1, no filtering is applied (self.fraction_filter = None, all samples are used)
            - If fraction is 0.5, half of the samples will be used for training and half for validation
        """
        self.__fraction = value
        if value == 1.0:
            self.__fraction_filter = None
        else:
            flips = np.random.uniform(0, 1, size=len(self.retrieval_index[self.__mode]))
            self.__fraction_filter = np.where(flips < value)[0]

    def use_fraction(self, fraction: float):
        self.fraction = fraction

    # endregion

    # region mode property

    ##################
    # Mode property to use only part of the dataset if it was split into train, validate, test or whatever parts
    ##################

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode: int):
        self.set_mode(mode, self.fraction)

    def set_mode(self, mode: int, fraction: float = 1.0):
        """Switching between train, validate and test parts if split was provided"""
        assert mode < len(
            self.__split_ratio
        ), f"Unable to set mode {mode}, not enough split parts with {self.__split_ratio}"
        self.__mode = mode
        self.fraction = fraction

    def train(self, fraction: float = 1.0):
        self.set_mode(0, fraction)

    def validate(self, fraction: float = 1.0):
        self.set_mode(1, fraction)

    valid = validate
    eval = validate

    def test(self, fraction: float = 1.0):
        self.set_mode(3, fraction)

    # endregion
