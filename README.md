Download cifar-10
```sh
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

Unzip it
```sh
tar -xf ./cifar-10-python.tar.gz
```

Initialize venv and activate it
```sh
python -m venv .
source bin/activate.fish # (for fish users; should be similar for other shells)
```

Install all the packages
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

Run
```sh
python src.py
```

Takes a **long** time to complete (like, several hours long)
