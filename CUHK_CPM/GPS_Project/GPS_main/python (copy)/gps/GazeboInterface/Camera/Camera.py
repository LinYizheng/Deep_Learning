#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    'CameraErr',
    'Camera',
]

import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))

# ROS topic
from Ros_Core import RosTopic as Ros
# ROS message
from sensor_msgs.msg import Image
# cv_bridge
from cv_bridge import CvBridge, CvBridgeError
# time
import time
# logging
import logging


class CameraErr(Exception):
    pass


class Camera(object):
    def __init__(self, rgbtopic=None, depthtopic=None):
        """
        :param rgbtopic: ros topic  to transport rgb data
        :param depthtopic: ros topic  to transport depth data
        """
        object.__init__(self)
        self.__Logger = logging.getLogger(self.__class__.__name__)
        # ros init
        if Ros.RosGetNode() is None:
            Ros.RosInit("GazeboInterface")
            self.Logger.info("ros node init!")
        self.__CvBridge = CvBridge()
        if rgbtopic is not None:
            self.__RGBData = None
            self.__WriteRgbFlag = False
            self.Logger.info("ros rgbtopic %s  init!" % (rgbtopic,))
            Ros.RosCallback(rgbtopic, self.__RgbCallback, Image)
        if depthtopic is not None:
            self.__DepthData = None
            self.__WriteDepthFlag = False
            self.Logger.info("ros depthtopic %s  init!" % (depthtopic,))
            Ros.RosCallback(rgbtopic, self.__DepthCallback, Image)

    @property
    def Logger(self):
        return self.__Logger

    def __RgbCallback(self, rgb):
        try:
            self.__RGBData = self.__CvBridge.imgmsg_to_cv2(rgb, "bgr8")
            self.__WriteRgbFlag = True
        except CvBridgeError as e:
            print e

    def __DepthCallback(self, depth):
        try:
            self.__DepthData = self.__CvBridge.imgmsg_to_cv2(depth, "passthrough")
            self.__WriteDepthFlag = True
        except CvBridgeError as e:
            print e

    def getRGBData(self):
        """
        :return: RGB data (np.array)
        """
        self.__WriteRgbFlag = False
        self.Logger.debug("Req Rgb data!")
        BeginTime = time.time()
        while not self.__WriteRgbFlag:
            if time.time() - BeginTime > 2.0:
                self.Logger.debug("time out in get rgb data!")
                return None
        self.Logger.debug("success in get rgb data!")
        return self.__RGBData

    def getDepthData(self):
        """
        :return: Depth data (np.array)
        """
        self.__WriteDepthFlag = False
        self.Logger.debug("Req Depth data!")
        BeginTime = time.time()
        while not self.__WriteDepthFlag:
            if time.time() - BeginTime > 2.0:
                self.Logger.debug("time out in get depth data!")
                return None
        self.Logger.debug("success in get depth data!")
        return self.__DepthData
