"""Dataset utilities used by the open-source MoLEx release."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

DEFAULT_LEN = 64600  # ~4 seconds @ 16 kHz
TARGET_SR = 16000

logger = logging.getLogger(__name__)


def load_audio(path: str) -> np.ndarray:
    """Load a mono waveform and resample it to 16 kHz."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def _repeat_or_trim(x: np.ndarray, max_len: int) -> np.ndarray:
    if len(x) >= max_len:
        return x[:max_len]
    reps = (max_len // len(x)) + 1
    return np.tile(x, reps)[:max_len]


def pad_random(x: np.ndarray, max_len: int = DEFAULT_LEN) -> np.ndarray:
    """Randomly crop/pad audio to a fixed length."""
    if len(x) > max_len:
        start = np.random.randint(0, len(x) - max_len + 1)
        return x[start:start + max_len]
    return _repeat_or_trim(x, max_len)


def pad_eval(x: np.ndarray, max_len: int = DEFAULT_LEN) -> np.ndarray:
    """Deterministically pad audio for evaluation."""
    return _repeat_or_trim(x, max_len)


def load_dictionary(file_name: str, delim: str = " ") -> Dict[str, str]:
    """Load a wav.scp-style mapping of utt_id -> path."""
    mapping = {}
    with open(file_name) as f:
        for line in f:
            parts = line.strip().split(delim, 1) if delim else line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
            mapping[key] = value.strip()
    return mapping


def gen_cyber_list(meta_file: Path, feat_file: Path) -> Tuple[List[str], List[int], List[str]]:
    """
    Filter a TSV metadata file using a wav.scp file.

    Returns:
        sample_keys: utterance IDs
        sample_labs: 1 for bonafide, 0 for spoof
        sample_paths: waveform paths aligned with IDs
    """
    scp_feats = load_dictionary(str(feat_file))
    fold_df = pd.read_csv(meta_file, sep="\t", header=0)
    fold_kv = dict(fold_df.values)

    missing = set(fold_kv) - set(scp_feats)
    if missing:
        logger.warning("No features for %d/%d files; removing.", len(missing), len(fold_kv))
        sample_keys = [k for k in fold_kv.keys() if k not in missing]
    else:
        sample_keys = list(fold_kv.keys())

    sample_labs = [1 if fold_kv[k] == "bonafide" else 0 for k in sample_keys]
    sample_paths = [scp_feats[k].replace('"', "") for k in sample_keys]
    logger.info("Loaded %d utterances from %s", len(sample_keys), meta_file)
    return sample_keys, sample_labs, sample_paths


class CyberDataset(Dataset):
    """Training dataset that randomly crops/pads audio segments."""

    def __init__(self, list_ids: Sequence[str], labels: Sequence[int], file_paths: Sequence[str]):
        assert len(list_ids) == len(labels) == len(file_paths)
        self.list_ids = list(list_ids)
        self.labels = list(labels)
        self.file_paths = list(file_paths)
        self.cut = DEFAULT_LEN

    def __len__(self) -> int:
        return len(self.list_ids)

    def __getitem__(self, index: int):
        utt_id = self.list_ids[index]
        path = self.file_paths[index]
        y = self.labels[index]

        audio = load_audio(path)
        audio = pad_random(audio, self.cut)
        return Tensor(audio), y, utt_id


class CyberEvalDataset(CyberDataset):
    """Evaluation dataset that uses deterministic padding."""

    def __getitem__(self, index: int):
        utt_id = self.list_ids[index]
        path = self.file_paths[index]
        y = self.labels[index]

        audio = load_audio(path)
        audio = pad_eval(audio, self.cut)
        return Tensor(audio), y, utt_id
