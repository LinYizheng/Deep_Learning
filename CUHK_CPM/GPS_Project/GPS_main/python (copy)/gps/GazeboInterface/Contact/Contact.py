#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    "Contact",
    "ContactErr"
]

# ROS topic
from Ros_Core import RosTopic as Ros
# Ros msg
from rr_plugin.msg import Contacts
# logging
import logging

class ContactErr(Exception):
    pass

class Contact(object):
    def __init__(self, WorldName, CallbackFunction):
        """
        :param WorldName: Gazebo WorldName
        :param CallbackFunction: function to receive contact message
        """
        object.__init__(self)
        self.__Logger = logging.getLogger(self.__class__.__name__)
        # ros init
        if Ros.RosGetNode() is None:
            Ros.RosInit("GazeboInterface")
            self.Logger.info("ros node init!")
        Ros.RosCallback("/" + WorldName + "/Contacts", CallbackFunction, Contacts)
        self.Logger.info("connect to /" + WorldName + "/Contacts!")

    @property
    def Logger(self):
        return self.__Logger
