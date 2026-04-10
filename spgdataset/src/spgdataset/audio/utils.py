import torch
import torch.nn.functional as F
import numpy as np
from soundfile import read as read_audio, write as write_audio
from subprocess import CalledProcessError, run
import numpy as np
import importlib.resources
import multiprocessing
from queue import Empty
from tqdm import tqdm
import os
import librosa
from math import ceil
import pickle

AUDIO_EXT = [
    "mp3",
    "wav",
    "aac",
    "ogg",
    "flac",
    "mp4",
    "mpeg",
    "mpga",
    "m4a",
    "webm",
]
# List of extensions here is just a primary filter. Since Soundfile is used
# to load_audio with ffmpeg as fallback, it really depends on what formats
# these two support.


def load_audio(filename, sr=32000, mono=True) -> np.ndarray:
    waveform2, _ = librosa.load(filename, sr=sr, mono=mono)
    return waveform2





def save_audio(filename: str, audio: np.ndarray, sr: int = 32000):
    write_audio(file=filename, data=audio, samplerate=sr)


def pad_or_trim(
    array: np.ndarray | torch.Tensor, length: int, mode="center"
) -> np.ndarray | torch.Tensor:
    assert len(array.shape) == 1, "Only mono audio is supported"
    if len(array) > length:
        array = array[:length]
    else:
        diff = length - len(array)
        if mode == "center":
            pw = diff // 2
        elif mode == "left":
            pw = 0
        else:
            pw = diff
        if torch.is_tensor(array):
            array = F.pad(array, (pw, diff - pw))
        else:
            array = np.pad(array, (pw, diff - pw))
    return array


def mel_filters(n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_64=librosa.filters.mel(sr=16000, n_fft=400, n_mels=64),
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {64, 80, 128}, f"Unsupported n_mels: {n_mels}"

    with importlib.resources.path("spgdataset.src.spgdataset.assets", "mel_filters.npz") as p:
        filters_path = p

    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"])


def log_mel_spectrogram(
    audio: str | np.ndarray | torch.Tensor,
    n_mels: int = 128,
    n_fft: int = 800,
    hop_length: int = 320,
) -> torch.Tensor:
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported right now

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """

    def convert(audio: str | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            return audio
        elif isinstance(audio, str):
            return torch.from_numpy(load_audio(audio))
        else:
            return torch.from_numpy(audio)

    data = convert(audio)

    window = torch.hann_window(n_fft)
    stft = torch.stft(data, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(n_mels).to(data.dtype)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def audio_load_length(filename: str, sr: int):
    audio = load_audio(filename, sr=sr)
    length = audio.shape[0]
    return length


def spectrogram_generator(
    data: dict,
    root: str,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    wsize: int,
    woffset: int,
    normalize: bool,
    dtype: torch.dtype,
) -> tuple[str, bool]:
    audio_filepath = data["file"]
    spg_filepath = data["cache_file"]
    audio = load_audio(audio_filepath, sr)
    audiot = torch.from_numpy(audio)
    original_size = audiot.size(0)
    if original_size <= wsize:
        audiot = pad_or_trim(array=audiot, length=wsize)
    else:
        num = ceil((original_size - wsize) / woffset)
        target_size = wsize + num * woffset
        assert wsize < target_size <= wsize + num * woffset
        audiot = pad_or_trim(array=audiot, length=target_size)
    spg = log_mel_spectrogram(
        audiot,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    ).to(dtype)
    if normalize is True:
        mean = torch.mean(spg)
        std = torch.std(spg)
        spg = (spg - mean) / std
    result = save_spectrogram(spg, spg_filepath, root)
    return (
        audio_filepath,
        result,
    )


def save_spectrogram(data: torch.Tensor, path: str, spectrograms_root: str) -> bool:
    def mkdir_blindly(path):
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except:
                pass

    if os.path.isfile(path):
        raise Exception(f"Cache file {path} already exists")
    mkdir_blindly(spectrograms_root)
    without_root = path.replace(spectrograms_root, "").lstrip("/")
    subdirs = without_root.split("/")[:-1]
    seq = spectrograms_root
    for subdir in subdirs:
        seq += "/" + subdir
        mkdir_blindly(seq)
    with open(path, "wb") as f:
        pickle.dump(data.cpu(), f, pickle.HIGHEST_PROTOCOL)
    return True


class MassProcessor:
    def __init__(
        self,
        handler_fn,
        settings: dict = {},
        num_workers: int = 0,
        progress_string: str = "",
    ) -> None:
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.handler_fn = handler_fn
        self.settings = settings
        self.num_workers = num_workers
        self.pstr = progress_string

    @staticmethod
    def worker(task_queue, result_queue, func, settings):
        while True:
            try:
                task_data = task_queue.get(timeout=1)
                if task_data is None:  # Exit signal
                    break
                index, task = task_data
                result = func(task, **settings)
                result_queue.put((index, result))
            except Empty:
                continue

    def __call__(self, tasks) -> dict:
        if self.num_workers == 0:
            results = {}
            for index, task in (pb := tqdm(enumerate(tasks), total=len(tasks))):
                pb.set_description_str(self.pstr)
                results[index] = self.handler_fn(task, **self.settings)
            return results
        # else multiprocessing
        for index, task in enumerate(tasks):
            self.task_queue.put((index, task))

        workers = [
            multiprocessing.Process(
                target=self.worker,
                args=(
                    self.task_queue,
                    self.result_queue,
                    self.handler_fn,
                    self.settings,
                ),
            )
            for _ in range(self.num_workers)
        ]
        for w in workers:
            w.start()
        for _ in range(len(workers)):
            self.task_queue.put(None)  # Send exit signal

        results = {index: None for index, _ in enumerate(tasks)}
        updates = 0
        with tqdm(total=len(tasks)) as pb:
            pb.set_description_str(self.pstr)
            while updates < len(tasks):
                try:
                    index, data = self.result_queue.get(timeout=1)
                    results[index] = data
                    pb.update(1)
                    updates += 1
                except Empty:
                    continue

        # Wait for all workers to finish
        for w in workers:
            w.join()
        return results
