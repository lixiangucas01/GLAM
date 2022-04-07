#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import numpy as np

np.random.seed(2021)
seeds = np.random.randint(0, 2**32 - 1, 100)

np.savetxt('seeds.txt',seeds,fmt='%d')
