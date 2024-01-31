# How to run this project

> PRE: Python version used 3.11.5


Obvioiusly these instructions are sound for a UNIX environment. Adapt to Windows if needed.


First, activate the virtual environment for the project and install all the required dependencies:

```shell
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Then, to run all tests:

```shell
pytest
```

Finally, to run the main script to train and assess the NN:

```shell
python -m nn.main
```
