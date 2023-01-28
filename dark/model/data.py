#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data related utility functions for working with DARK models
"""

import numpy as np

def seq2aa(seq):
    abc=np.array(list("ARNDCQEGHILKMFPSTWYV"))
    return("".join(list(abc[seq])))
