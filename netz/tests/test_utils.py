# -*- coding: utf-8 -*-
from __future__ import division

import pytest

from ..utils import flatten


@pytest.mark.parametrize('lst, expected', [
    ([], []),
    ([[]], []),
    ([(1, 2), (3, 4)], [(1, 2), (3, 4)]),
    ([[], (1, 2), []], [(1, 2)]),
    ([[], (1, 2), [], [(3, 4)]], [(1, 2), (3, 4)]),
    ([[], (1, 2), [], [[(3, 4)]]], [(1, 2), [(3, 4)]]),
])
def test_flatten(lst, expected):
    result = flatten(lst)
    assert result == expected
