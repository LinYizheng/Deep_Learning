#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    'GripperErr',
    'Gripper',
]

import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
from Ros_Core.RosServer import GetGripperDataReq, SetGripperDataReq
import logging


class GripperErr(Exception):
    pass


class Gripper(object):
    def __init__(self, ModelName):
        """
        :param ModelName: Gripper parent model Name
        """
        object.__init__(self)
        self.__Logger = logging.getLogger(self.__class__.__name__)
        self.ClosePosition = -0.0315
        self.OpenPosition = 0
        self.SetGripperServerName = "/" + ModelName + "/SetGripperPos"
        self.GetGripperServerName = "/" + ModelName + "/GetGripperPos"

    @property
    def Logger(self):
        return self.__Logger

    def catch(self):
        """
        :return: if gripper close success (True or False)
        """
        self.Logger.debug("gripper catch!")
        res = SetGripperDataReq(self.SetGripperServerName, self.ClosePosition, self.ClosePosition)
        return res.success

    def release(self):
        """
        :return: if gripper open success (True or False)
        """
        self.Logger.debug("gripper release!")
        res = SetGripperDataReq(self.SetGripperServerName, self.OpenPosition, self.OpenPosition)
        return res.success

    def setFingerPos(self, LeftFingerPos, RightFingerPos):
        """
        :param LeftFingerPos: left finger pos
        :param RightFingerPos: right finger pos
        :return: if gripper open success (True or False)
        """
        self.Logger.debug("gripper set position left: %s, right: %s!" % (LeftFingerPos, RightFingerPos,))
        res = SetGripperDataReq(self.SetGripperServerName, LeftFingerPos, RightFingerPos)
        return res.success

    def getFingerPos(self, isDegree=False):
        """
        :param isDegree: if use degree or rad, True: use Degree, False: use Rad.
        :return: left finger pos, right finger pos
        """
        res = GetGripperDataReq(self.GetGripperServerName, isDegree)
        self.Logger.debug("gripper get position left: %s, right: %s!" % (res.leftfingerpos, res.rightfingerpos,))
        return res.leftfingerpos, res.rightfingerpos

    def isCmdMovingDone(self, command):
        """

        :param command: 'catch' or 'release'
        :return: is command is run finished False or True
        """
        if str.upper(command) == "CATCH":
            return self.isPosMovingDone(self.ClosePosition, self.ClosePosition)
        elif str.upper(command) == "RELEASE":
            return self.isPosMovingDone(self.OpenPosition, self.OpenPosition)
        else:
            print "command should be 'catch' or 'release'"
            return False

    def isPosMovingDone(self, left_pos, right_pos):
        """
        :param left_pos:
        :param right_pos:
        :return:
        """
        FbkPos = self.getFingerPos(isDegree=False)
        if abs(FbkPos[0] - left_pos) < 2e-3 and abs(FbkPos[1] - right_pos) < 2e-3:
            return True
        else:
            return False
