# Environment Setup


- We use Anaconda to install required packages:
    ```bash
        conda env create -f env/env_equfiformer.yml
    ```
    This will create a new environment called `equiformer`.

- We activate the environment:
    ```bash
        export PYTHONNOUSERSITE=True    # prevent using packages from base
        conda activate equiformer
    ```

- Besides, [`env/env_equiformer.yml`](../env/env_equiformer.yml) specifies versions of all packages.

- After setting up the environment, clone OC20 repository and install `ocpmodels`:
    ```bash
        git clone https://github.com/Open-Catalyst-Project/ocp
        cd ocp
        git checkout b5a197f
        pip install -e .
    ```
    The version of `ocpmodels` used here is `0.0.3`. 
    The correpsonding version of GitHub repository is [here](https://github.com/Open-Catalyst-Project/ocp/tree/b5a197fc3c79a9a5a787aabaa02979be53d296b7).
