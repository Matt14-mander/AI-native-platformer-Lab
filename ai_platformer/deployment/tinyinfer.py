"""Persistent ctypes adapter for the local TinyInfer bridge; no Torch/SB3 imports."""

from __future__ import annotations

import ctypes
from pathlib import Path
from threading import RLock

import numpy as np

from .package import load_actor_metadata


class TinyInferPolicy:
    def __init__(self, model_path: Path, library_path: Path, *, fuse_relu: bool = False):
        self.metadata = load_actor_metadata(Path(model_path))
        self.input_size = self.metadata["input"]["shape"][1]
        self.output_size = self.metadata["output"]["shape"][1]
        self._lock = RLock()
        self._handle = None
        self._library = ctypes.CDLL(str(Path(library_path).resolve()))
        lib = self._library
        float_pointer = ctypes.POINTER(ctypes.c_float)
        lib.pti_abi_version.argtypes = []
        lib.pti_abi_version.restype = ctypes.c_int
        lib.pti_last_error.argtypes = []
        lib.pti_last_error.restype = ctypes.c_char_p
        lib.pti_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.pti_create.restype = ctypes.c_void_p
        lib.pti_infer.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            ctypes.c_size_t,
            float_pointer,
            ctypes.c_size_t,
        ]
        lib.pti_infer.restype = ctypes.c_int
        lib.pti_destroy.argtypes = [ctypes.c_void_p]
        lib.pti_destroy.restype = None
        if lib.pti_abi_version() != 1:
            raise ValueError("unsupported TinyInfer bridge ABI")
        self._handle = lib.pti_create(
            str(Path(model_path).resolve()).encode("utf-8"),
            self.metadata["input"]["name"].encode("utf-8"),
            self.metadata["output"]["name"].encode("utf-8"),
            self.input_size,
            self.output_size,
            int(fuse_relu),
        )
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        return self._library.pti_last_error().decode("utf-8", errors="replace")

    def logits(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation)
        if values.dtype != np.float32 or values.shape not in {
            (self.input_size,),
            (1, self.input_size),
        }:
            raise ValueError(f"expected float32 [{self.input_size}] or [1,{self.input_size}]")
        if not np.isfinite(values).all():
            raise ValueError("observation must be finite")
        values = np.ascontiguousarray(values)
        output = np.empty(self.output_size, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        with self._lock:
            if not self._handle:
                raise RuntimeError("TinyInfer policy is closed")
            status = self._library.pti_infer(
                self._handle,
                values.ctypes.data_as(pointer),
                values.size,
                output.ctypes.data_as(pointer),
                output.size,
            )
            if status != 0:
                raise RuntimeError(self._error())
        if not np.isfinite(output).all():
            raise RuntimeError("TinyInfer returned non-finite logits")
        return output

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        if not deterministic:
            raise ValueError("TinyInfer deployment currently supports deterministic actions only")
        return np.asarray(self.logits(observation).argmax(), dtype=np.int64), None

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._library.pti_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()
