#!/usr/bin/env python
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    "RosCallback",
    "RosPublish",
    "RosInit",
]
# ROS lib
import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
import rospy


def RosCallback(topic, function, DataType):
    RosSub = rospy.Subscriber(topic, DataType, function)
    return RosSub


def RosPublish(topic, DataType):
    RosPub = rospy.Publisher(topic, DataType, queue_size=10)
    return RosPub


def RosInit(RosNodeName):
    rospy.init_node(str(RosNodeName), anonymous=True)


def RosGetNode():
    return rospy.get_node_uri()
