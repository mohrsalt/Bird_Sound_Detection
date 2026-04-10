from spgdataset.src.spgdataset.dataset import SpectrogramDataset
import numpy as np
import logging

####################################################################
#                 Spectrogram Dataset Router
####################################################################
# Router class, allows to access multiple SpectrogramDatasets
# simultaneously.


class SpectrogramDatasetRouter:
    def __init__(self, datasets: list[SpectrogramDataset]) -> None:
        """
        Router class, allows to access multiple SpectrogramDatasets
        simultaneously.
        Args:
            datasets (list[SpectrogramDataset]): list of SpectrogramDatasets
        """
        self.datasets = datasets
        self.output_configuration = self.datasets[0].get_output_configuration()
        # assert all datasets have same output configuration
        assert all(
            oc == self.output_configuration
            for oc in [ds.get_output_configuration() for ds in self.datasets]
        ), "Output configurations of all datasets should match"
        # assert that all datasets have the same mode
        if not all(ds.mode == datasets[0].mode for ds in datasets):
            logging.warning(
                "Datasets have different modes, switching to mode 0 (train)"
            )
            self.train()

    @property
    def classes(self):
        self._classes = [c for ds in self.datasets for c in ds.classes]
        return self._classes

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        self._datasets = list(value)
        self.update_lengths()

    def train(self, fraction: float = 1.0):
        self._set_mode("train", fraction=fraction)

    def validate(self, fraction: float = 1.0):
        self._set_mode("validate", fraction=fraction)

    valid = validate
    eval = validate

    def test(self, fraction: float = 1.0):
        self._set_mode("test", fraction=fraction)

    def update_lengths(self):
        self._lengths = [0] + list(np.cumsum([len(ds) for ds in self._datasets]))

    def _set_mode(self, mode: str, fraction: float = 1.0):
        for dataset in self.datasets:
            if mode == "train":
                dataset.train(fraction=fraction)
            elif mode == "validate":
                dataset.validate(fraction=fraction)
            elif mode == "test":
                dataset.test(fraction=fraction)
        self.update_lengths()

    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def __getitem__(self, index):
        dnum, ix = self._get_coords(index)
        output = self.datasets[dnum].__getitem__(ix)
        if dnum > 0 and "label" in output:
            local_class_index = output["label"]
            class_count = [len(ds.classes) for ds in self.datasets]
            global_class_index = (
                sum([el for el in class_count[:dnum]]) + local_class_index
            )
            output["label"] = global_class_index
        return output

    def _get_coords(self, index):
        for i in range(len(self._lengths) - 1):
            if index >= self._lengths[i] and index < self._lengths[i + 1]:
                return i, index - self._lengths[i]
        raise Exception("Index out of bounds")
