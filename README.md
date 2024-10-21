# Get and format HIGGS dataset

Get the dataset
```sh
# I would have put a wget command here, but the dataset itself seems to not be available anymore

# Following instructions assume you have the dataset at `./HIGGS.csv.gz`
```

Create and enter the venv
```sh
python -m venv .
source bin/activate.fish # for fish users; should similar to other shells
```

Recompress the dataset (from compressed csv to binary)
```sh
mkdir higgs
python recompress.py
```

Filter & regularize dataset

I uses Rust for that, since Python used more ram than was available to my laptop.

[Install Rust](https://doc.rust-lang.org/cargo/getting-started/installation.html)
```sh
cd regularize
cargo r --release
```
