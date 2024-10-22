Currently-committed code solved following task:
$$
\begin{cases}
u(x, y) \\
r = \sqrt{x^2 + y^2} \in [0; 4]\\
u_{xx} + u_{yy} = f(x, y) \\
u(\{r = 4\})_{\varphi} = 0 \\
u(-1, 0) = 0\\
\end{cases}
$$

# How to run

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
