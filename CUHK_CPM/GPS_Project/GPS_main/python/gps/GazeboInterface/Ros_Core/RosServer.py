#!/usr/bin/env python
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    "SetModelStateReq",
    "GetModelStateReq",
    "SetLinkStateReq",
    "GetLinkStateReq",
    "SetFloat64MultiArrayReq",
    "GetGripperDataReq",
    "SetGripperDataReq",
    "SetSpawnModelXmlReq",
    "SetDeleteModelReq",
    "SetBoolReq"

]
# ROS lib
import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from std_srvs.srv import SetBool
from rr_plugin.srv import SetFloat64MultiArray
from rr_plugin.srv import GetGripperData
from rr_plugin.srv import SetGripperData
from rr_plugin.srv import AddGroup
from rr_plugin.srv import DelGroup
from rr_plugin.srv import SetPoint


def SetModelStateReq(model_state, RosServerName='/gazebo/set_model_state'):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, SetModelState)
        res = req(model_state)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def GetModelStateReq(model_name, relative_entity_name, RosServerName='/gazebo/get_model_state'):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, GetModelState)
        res = req(model_name, relative_entity_name)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def SetLinkStateReq(link_state, RosServerName='/gazebo/set_link_state'):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, SetLinkState)
        res = req(link_state)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def GetLinkStateReq(model_name, relative_entity_name, RosServerName='/gazebo/get_link_state'):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, GetLinkState)
        res = req(model_name, relative_entity_name)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def SetFloat64MultiArrayReq(RosServerName, data):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, SetFloat64MultiArray)
        res = req(data)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def GetGripperDataReq(RosServerName, isDegree):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, GetGripperData)
        res = req(isDegree)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def SetGripperDataReq(RosServerName, leftpos, rightpos):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, SetGripperData)
        res = req(leftpos, rightpos)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def SetSpawnModelXmlReq(model_name, model_xml, initial_pose, reference_frame="world",
                      RosServerName="/gazebo/spawn_sdf_model"):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, SpawnModel)
        res = req(model_name, model_xml, "", initial_pose, reference_frame)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def SetDeleteModelReq(model_name, RosServerName="/gazebo/delete_model"):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, DeleteModel)
        res = req(model_name)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

def SetBoolReq(Bool, RosServerName):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, SetBool)
        res = req(Bool)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

def AddGroupReq(type, color, RosServerName):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, AddGroup)
        res = req(type, color)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

def DelGroupReq(type, index, RosServerName):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, DelGroup)
        res = req(type, index)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

def SetPointReq(type, index, point, RosServerName):
    rospy.wait_for_service(RosServerName)
    try:
        req = rospy.ServiceProxy(RosServerName, SetPoint)
        res = req(type, index, point)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e
