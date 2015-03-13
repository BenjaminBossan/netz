# -*- coding: utf-8 -*-
from __future__ import division

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")


def pytest_runtest_setup(item):
    if 'skipthis' in item.keywords:
        pytest.skip("This is not fit for test.")
