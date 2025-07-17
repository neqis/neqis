# NeQIS


This is the implementation of **NeQIS: Neural Quadratic Implicit Surfaces**.

### [Project page](https://neqis.github.io/)

----------------------------------------
## Installation

```shell
git clone https://github.com/neqis/neqis.git
cd neqis
pip install -r requirements.txt
```

## Usage

#### Data Convention

Our data format is inspired from [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md) as follows:
```
CASE_NAME
|-- cameras.npz    # camera parameters
|-- image
    |-- 000.png        # image for each view
    |-- 001.png
    ...
|-- normal
    |-- 000.png        # normal map for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # mask for each view
    |-- 001.png
    ...
```

One can create folders with different data in it, for instance, a normal folder for each normal estimation method.
The name of the folder must be set in the used `.conf` file.

### Run

**Train**

```shell
python run_experiment.py --mode train --conf ./confs/CONF_NAME.conf --case CASE_NAME
```
