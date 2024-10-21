import csv
import gzip
import struct
from typing import BinaryIO, Iterator, List


def recompress(fin, f_sig, f_nse):
    r: Iterator[List[str]] = csv.reader(fin)
    while (entry := next(r, None)) is not None:  # type:ignore
        f_out: BinaryIO = f_sig if float(entry[0]) == 1.0 else f_nse
        entry_data = list(map(float, entry[1:]))
        entry_bytes = struct.pack("=" + "f" * len(entry_data), *entry_data)
        f_out.write(entry_bytes)


with gzip.open("HIGGS.csv.gz", mode="rt") as fin:
    with open("./higgs/signal", mode="wb") as f_sig:
        with open("./higgs/noise", mode="wb") as f_nse:
            recompress(fin, f_sig, f_nse)
