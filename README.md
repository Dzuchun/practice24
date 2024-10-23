# Get and format HIGGS dataset

Get the dataset
```sh
wget https://archive.ics.uci.edu/static/public/280/higgs.zip
unzip higgs.zip
# note: this takes a loooong time, it's like 10GB
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

# Run the model

Install the dependencies
```sh
pip install -r requirements.txt
```

Run the model
```sh
python src.py
```
