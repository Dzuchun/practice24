import struct
from sys import stderr
from typing import Iterator, List

PARAMS_PER_ENTRY = 21
ENTRY_DATA_SIZE = PARAMS_PER_ENTRY * 4


def _load_data(fname, entries, skip) -> Iterator[List[float]]:
    with open(fname, mode="rb") as file:
        for _ in range(skip):
            _ = file.read(ENTRY_DATA_SIZE)
        for _ in range(entries):
            entry_data = file.read(ENTRY_DATA_SIZE)
            if len(entry_data) != ENTRY_DATA_SIZE:
                # that's the end of the file
                print(
                    "WARNING: hit EOF after reading less entries than requested",
                    file=stderr,
                )
                break
            yield list(struct.unpack("=" + "f" * PARAMS_PER_ENTRY, entry_data))


def load_signal(entries=100_000, skip=0) -> Iterator[List[float]]:
    return _load_data("./higgs/signal_reg", entries, skip)


def load_noise(entries=100_000, skip=0) -> Iterator[List[float]]:
    return _load_data("./higgs/noise_reg", entries, skip)
