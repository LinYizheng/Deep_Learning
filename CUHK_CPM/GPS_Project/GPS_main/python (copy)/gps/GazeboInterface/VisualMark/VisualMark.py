#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    'VisualMarkErr',
    'VisualMark',
]

import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
from Ros_Core.RosServer import AddGroupReq, DelGroupReq, SetPointReq
from geometry_msgs.msg import Point
import logging


class VisualMarkErr(Exception):
    pass


class VisualMark(object):
    def __init__(self):
        object.__init__(self)
        self.__Logger = logging.getLogger(self.__class__.__name__)
        self.AddGroupServerName = "/visual/AddGroup"
        self.DelGroupServerName = "/visual/DelGroup"
        self.SetPointServerName = "/visual/SetPoint"

    def addGroup(self, type, color):
        """

        :param type: 'POINT' or 'LINE'
        :param color: 'Red', 'Green', 'Blue', 'Grey', .......
        :return: if setting success
        """
        res = AddGroupReq(str.upper(type), "Gazebo/" + str.upper(color)[0] + str.lower(color)[1:],
                          self.AddGroupServerName)
        return res.success

    def delGroup(self, type, index):
        """

        :param type: 'POINT' or 'LINE'
        :param index: a index to group_point or group_line
        :return: if setting success
        """
        res = DelGroupReq(str.upper(type), index, self.DelGroupServerName)
        return res.success

    def setPoint(self, type, index, pos):
        """

        :param type:
        :param index:
        :param pos:
        :return:
        """
        assert len(pos) == 3, 'pos required 3'
        point = Point()
        point.x = pos[0] / 1000.0
        point.y = pos[1] / 1000.0
        point.z = pos[2] / 1000.0
        res = SetPointReq(str.upper(type), index, point, self.SetPointServerName)
