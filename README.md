Download the dataset
```sh
wget https://zenodo.org/records/8370883/files/BNB_All_NoWire_00.h5
```

All of the following programs require Rust toolchain for be compiled. Toolchain installation instructions can be found [here](https://www.rust-lang.org/tools/install).

Run the example
```sh
cargo r --release
```
Note: this example takes **a lot** of RAM. Can't reasonably do anything about it - it's just that many data to keep.
