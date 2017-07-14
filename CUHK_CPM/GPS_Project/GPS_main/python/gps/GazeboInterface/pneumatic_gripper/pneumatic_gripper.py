#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    'PneumaticGripperErr',
    'PneumaticGripper',
]

import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
from Ros_Core.RosServer import SetBoolReq
from Ros_Core.RosTopic import RosCallback, RosGetNode, RosInit
from std_msgs.msg import Bool
import logging

class PneumaticGripperErr(Exception):
    pass

class PneumaticGripper(object):
    def __init__(self, ModelName):
        """
        :param ModelName: Gripper parent model Name
        """
        object.__init__(self)
        self.GripperState = False
        self.__Logger = logging.getLogger(self.__class__.__name__)
        # ros init
        if RosGetNode() is None:
            RosInit("GazeboInterface")
            self.Logger.info("ros node init!")
        self.CallBackTopic = '/' + ModelName + '/pneumatic_gripper_state'
        self.ControlServer = '/' + ModelName + '/pneumatic_gripper_control'
        RosCallback(self.CallBackTopic, self.__GripperCallback, Bool)
    @property
    def Logger(self):
        return self.__Logger

    def __GripperCallback(self, state):
        self.GripperState = state.data

    def catch(self):
        self.Logger.debug("gripper catch!")
        res = SetBoolReq(True, self.ControlServer,)
        return res.success

    def release(self):
        self.Logger.debug("gripper release!")
        res = SetBoolReq(False, self.ControlServer)
        return res.success

    def getGripperState(self):
        return self.GripperState


