
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings(action='ignore')

import tensorflow as tf
import numpy as np
import math
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

parser=argparse.ArgumentParser(prog='PROG', formatter_class=argparse.RawDescriptionHelpFormatter,
description='''\
Prediction of the pressure loss coefficient of 90degree bends.
This code is based on the Python 3.6.8 version and Tensorflow 1.12.0 version.''')


parser.add_argument('-d','--diameter',metavar='', required=True, type=float, help='Diameter of pipe [m]')
parser.add_argument('-R','--radius',metavar='', required=True, type=float, help='Radius of curvature [m]')
parser.add_argument('-rho','--density',metavar='', required=True, type=float, help='Density [kg/m3]')
parser.add_argument('-mu','--viscosity',metavar='', required=True, type=float, help='Dynamic  Viscosity [kg/ms]')
parser.add_argument('-Q','--flowrate', metavar='', required=True, type=float, help='Volume flow rate [m3/s]')

args=parser.parse_args()

X=tf.placeholder(tf.float32 , shape=[None,2])
Y=tf.placeholder(tf.float32 , shape=[None,1])

w1=tf.get_variable("w1", shape=[2,20],initializer=tf.contrib.layers.variance_scaling_initializer())
b1=tf.Variable(tf.random_uniform([1], minval=0., maxval=1), name='bias1')
layer1=tf.sigmoid(tf.matmul(X,w1)+b1)


w5=tf.get_variable("w5", shape=[20,1],initializer=tf.contrib.layers.variance_scaling_initializer())
b5=tf.Variable(tf.random_uniform([1], minval=0., maxval=1), name='bias5')
hypo=tf.matmul(layer1,w5)+b5

cost=tf.reduce_mean(tf.square(hypo-Y))

saver = tf.train.Saver()


# In[2]:


save_file = './K_90bend.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)
    
    velocity = (4*args.flowrate)/(args.diameter*args.diameter*math.pi)

    Velocity = round(velocity,2)

    Re = (args.density*Velocity*args.diameter)/args.viscosity

    PIRNT_Re=round(Re,2)

    Re_log = math.log10(Re)

    Rr = args.radius/(args.diameter/2)

    PIRNT_Rr = round(Rr,2)

    feed_input = np.array([[Re_log,Rr]])

    prediction = sess.run(hypo, feed_dict={X : feed_input})

    prediction_result = prediction[0,0]
    
    pressure_drop = prediction_result*args.density*0.5*Velocity*Velocity

    Pressure_drop = round(pressure_drop,2)
   
    print("---Prediction of pressure loss coefficient---\n")
    print("Prsssure loss coefficient = %f \n\n" %prediction_result)
    print("---------Calculation of pressure drop--------\n\n")
    print("Reynolds number:%f \n" %PIRNT_Re)
    print("Velocity (m/s):%f \n" %Velocity)
    print("Pressure drop (Pa): %f" %Pressure_drop)

