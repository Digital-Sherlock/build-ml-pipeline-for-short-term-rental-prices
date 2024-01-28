'''
conftest.py contains pytest fixtures for
peforming data checks. In addition, it holds
settings for defining command-line arguments for
pytest.
'''

import pytest
import pandas as pd
import wandb

# Configuring CLI arguments for pytest
def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


# Session fixtures
@pytest.fixture(scope='session')
def data(request):
    '''
    Download dataset from W&B.

    - Input:
        - request: access for pytest CLI args
    - Output:
        - df: (pd.DataFrame) Pandas DF
    '''

    run = wandb.init(job_type="data_tests")

    # Accessing command-line arguments
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    '''
    Download reference dataset from W&B.

    - Input:
        - request: access for pytest CLI args
    - Output:
        - df: (pd.DataFrame) Pandas DF
    '''
        
    run = wandb.init(job_type="data_tests", resume=True)

    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    '''
    Read kl_threshold argument from CLI.

    - Input:
        - request: access for pytest CLI args
    - Output:
        - kl_threshold: (float) kl_threshold
    '''
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    '''
    Read min_price argument from CLI.

    - Input:
        - request: access for pytest CLI args
    - Output:
        - min_price: (float) min_price
    '''
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    '''
    Read max_price argument from CLI.

    - Input:
        - request: access for pytest CLI args
    - Output:
        - max_price: (float) max_price
    '''
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)