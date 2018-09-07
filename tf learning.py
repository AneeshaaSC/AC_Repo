# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:35:13 2017

@author: 212458792
"""

import tensorflow as tf

a=tf.ones([1, 4, 3,2])
b=a.get_shape().as_list()
    
#print(b.eval())

sess=tf.Session()
r=sess.run(b)

print(r)

