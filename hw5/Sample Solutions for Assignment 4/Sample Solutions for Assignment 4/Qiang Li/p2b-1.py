# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:08:50 2018

@author: qli14
"""
import numpy as np
a = 0.0
b = 0.0

for i in range(0,50):
    print("Step ", i)
    a_temp = 1 + 3.0 / 4.0 * ( 1.0/3.0 * a + 2.0/3.0 * b )
    b = 2 + 3.0 / 4.0 * ( 2.0/3.0 * a + 1.0/3.0 * b )
    a = a_temp
    print("a = %0.3f, b = %0.3f" % (a, b) )
