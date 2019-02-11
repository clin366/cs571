# ========================================================================
# Copyright 2019 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import csv
import os
import pytest

__author__ = "Gary Lai"


def dataset():
    res_dir = os.environ.get('RESOURCE')
    with open(os.path.join(res_dir, 'hashtags.csv'), 'r') as fin:
        reader = csv.reader(fin)
        return [row for row in reader if row]


@pytest.mark.parametrize('hashtag, expected', dataset())
def test_hello_word(seg, hashtag, expected):
    assert ' '.join(seg.decode(hashtag)) == expected
