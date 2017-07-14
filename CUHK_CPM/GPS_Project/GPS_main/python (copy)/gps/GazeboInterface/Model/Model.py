#!/usr/bin/env python
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    'Model',
    "ModelErr"
]

import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))

from Ros_Core.RosServer import *
from gazebo_msgs.msg import LinkState
from geometry_msgs.msg import Pose
import numpy as np
import tf.transformations as tf
from ModelMaker import *
import logging


class ModelErr(Exception):
    pass


class Model(object):
    def __init__(self):
        object.__init__(self)
        self.__Logger = logging.getLogger(self.__class__.__name__)
        self.Logger.info("Create Model module!")

    @property
    def Logger(self):
        return self.__Logger

    def setLinkPos(self, link_name, position, reference_frame="world"):
        """
        :param link_name: Model name which you want to set pos, eg: 'table'
        :param position: Position you want to set. np.array([x_mm, y_mm, z_mm, u_deg, v_deg, w_deg])
        :param reference_frame: Reference rotation frame, default to "world"
        :return:
        """
        self.Logger.debug("model: %s set to %s reference frame is %s." % (link_name, position, reference_frame,))
        OriAngle = tf.quaternion_from_euler(np.deg2rad(position[5]), np.deg2rad(position[4]), np.deg2rad(position[3]))
        PosState = LinkState()
        PosState.link_name = link_name
        PosState.pose.position.x = position[0] / 1000.0
        PosState.pose.position.y = position[1] / 1000.0
        PosState.pose.position.z = position[2] / 1000.0
        PosState.pose.orientation.x = OriAngle[0]
        PosState.pose.orientation.y = OriAngle[1]
        PosState.pose.orientation.z = OriAngle[2]
        PosState.pose.orientation.w = OriAngle[3]
        PosState.twist.linear.x = 0.0
        PosState.twist.linear.y = 0.0
        PosState.twist.linear.z = 0.0
        PosState.twist.angular.x = 0.0
        PosState.twist.angular.y = 0.0
        PosState.twist.angular.z = 0.0
        PosState.reference_frame = reference_frame
        res = SetLinkStateReq(PosState)
        return res.success

    def getLinkPos(self, link_name, reference_frame="world"):
        """
        :param link_name: Model name which you want to get pos, eg: 'table'
        :param reference_frame: Reference rotation frame, default to "world"
        :return: model position you get.  np.array([x_mm, y_mm, z_mm, u_deg, v_deg, w_deg])
        """
        res = GetLinkStateReq(link_name, reference_frame)
        quaternion = (
            res.link_state.pose.orientation.x,
            res.link_state.pose.orientation.y,
            res.link_state.pose.orientation.z,
            res.link_state.pose.orientation.w

        )
        EulaAngle = tf.euler_from_quaternion(quaternion)
        LinkPos = np.array([res.link_state.pose.position.x * 1000.0, res.link_state.pose.position.y * 1000.0,
                            res.link_state.pose.position.z * 1000.0, np.rad2deg(EulaAngle[2]), np.rad2deg(EulaAngle[1]),
                            np.rad2deg(EulaAngle[0])])
        self.Logger.debug(
            'model: %s position is %s,  reference frame is %s.' % (link_name, LinkPos, reference_frame,))
        return LinkPos

    def createBoxModel(self, mass, size, color, static=False):
        """
        :param mass: The mass of model
        :param size: Box size. eg. [x_m, y_m, z_m]
        :param color: 'Grey", "DarkGrey", "White", "FlatBlack", "Black", "Red", "RedBright", "Green", "Blue", "SkyBlue","Yellow", "ZincYellow", "DarkYellow", "Purple", "Orange",if you want model to be transparent, add "transparent" end of color, eg."GreyTransparent"
        :param static: if model be static, not affected by gravity.True: not be affected by gravity.False: be affected by gravity.default to False.
        :return: The description of box model, xml
        """
        self.Logger.debug("Create Box Model!")
        xml = CreateBoxModel(mass, size[0], size[1], size[2], color, static)
        return xml

    def createCylinderModel(self, mass, radius, length, color, static=False):
        """
        :param mass: The mass of model
        :param radius: The radius of Cylinder
        :param length: The length of Cylinder
        :param color:  'Grey", "DarkGrey", "White", "FlatBlack", "Black", "Red", "RedBright", "Green", "Blue", "SkyBlue","Yellow", "ZincYellow", "DarkYellow", "Purple", "Orange",if you want model to be transparent, add "transparent" end of color, eg."GreyTransparent"
        :param static:  if model be static, not affected by gravity.True: not be affected by gravity.False: be affected by gravity.default to False.
        :return: The description of box model, xml
        """
        self.Logger.debug("Create Cylinder Model!")
        xml = CreateCylinderModel(mass, radius, length, color, static)
        return xml

    def createSphereModel(self, mass, radius, color, static=False):
        """
        :param mass: The mass of model.
        :param radius: The radius of Sphere.
        :param color: 'Grey", "DarkGrey", "White", "FlatBlack", "Black", "Red", "RedBright", "Green", "Blue", "SkyBlue", "Yellow", "ZincYellow", "DarkYellow", "Purple", "Orange",if you want model to be transparent, add "transparent" end of color, eg."GreyTransparent"
        :param static: if model be static, not affected by gravity.True: not be affected by gravity.False: be affected by gravity.default to False.
        :return: The description of box model xml.
        """
        self.Logger.debug("Create Sphere Model!")
        xml = CreateSphereModel(mass, radius, color, static)
        return xml

    def createBoxVisual(self, size, color):
        """
        :param size: Box size. eg. [x_m, y_m, z_m]
        :param color: 'Grey", "DarkGrey", "White", "FlatBlack", "Black", "Red", "RedBright", "Green", "Blue", "SkyBlue","Yellow", "ZincYellow", "DarkYellow", "Purple", "Orange",if you want model to be transparent, add "transparent" end of color, eg."GreyTransparent"
        :return: The description of box model, xml
        """
        self.Logger.debug("Create Box Visual!")
        xml = CreateBoxVisual(size[0], size[1], size[2], color)
        return xml

    def createCylinderVisual(self, radius, length, color):
        """
        :param radius: The radius of Cylinder
        :param length: The length of Cylinder
        :param color: 'Grey", "DarkGrey", "White", "FlatBlack", "Black", "Red", "RedBright", "Green", "Blue", "SkyBlue","Yellow", "ZincYellow", "DarkYellow", "Purple", "Orange",if you want model to be transparent, add "transparent" end of color, eg."GreyTransparent"
        :return: The description of Cylinder visual, xml
        """
        self.Logger.debug("Create Cylinder Visual!")
        xml = CreateCylinderVisual(radius, length, color)
        return xml

    def createSphereVisual(self, radius, color):
        """
        :param radius: The radius of Sphere
        :param color: 'Grey", "DarkGrey", "White", "FlatBlack", "Black", "Red", "RedBright", "Green", "Blue", "SkyBlue","Yellow", "ZincYellow", "DarkYellow", "Purple", "Orange",if you want model to be transparent, add "transparent" end of color, eg."GreyTransparent"
        :return: The description of Sphere visual, xml
        """
        self.Logger.debug("Create Sphere Visual!")
        xml = CreateSphereVisual(radius, color)
        return xml

    def addModelXml(self, model_name, model_xml, init_pos=None, reference_frame="world"):
        """
        :param model_name: The model name you want to add in gazebo
        :param model_xml: Description of model, xml
        :param init_pos: Model initial pos when model spawn, np.array([x_mm, y_mm, z_mm, u_deg, v_deg, w_deg])
        :param reference_frame: Reference rotation frame, default to "world"
        :return: if spawn success
        """
        if init_pos is None:
            init_pos = [0, 0, 0, 0, 0, 0]
        self.Logger.debug("Add model %s in %s." % (model_name, init_pos,))
        OriAngle = tf.quaternion_from_euler(np.deg2rad(init_pos[5]), np.deg2rad(init_pos[4]), np.deg2rad(init_pos[3]))
        initial_pose = Pose()
        initial_pose.position.x = init_pos[0] / 1000.0
        initial_pose.position.y = init_pos[1] / 1000.0
        initial_pose.position.z = init_pos[2] / 1000.0
        initial_pose.orientation.x = OriAngle[0]
        initial_pose.orientation.y = OriAngle[1]
        initial_pose.orientation.z = OriAngle[2]
        initial_pose.orientation.w = OriAngle[3]
        res = SetSpawnModelXmlReq(model_name, model_xml, initial_pose, reference_frame)
        return res.success

    def addGazeboModel(self, model_name, model_type, init_pos=None, reference_frame="world"):
        """

        :param model_name: The model name you want to add in gazebo
        :param model_type: in ~/.gazebo/models/ model folder name
        :param init_pos: Model initial pos when model spawn, np.array([x_mm, y_mm, z_mm, u_deg, v_deg, w_deg])
        :param reference_frame: if spawn success
        :return:
        """
        with open("/home/pi/.gazebo/models/" + str(model_type) + "/model.sdf", "r") as f:
            model_xml = f.read()
        assert model_xml != None, "~/.gazebo/models/" + str(model_type) + "/model.sdf, not data."
        if init_pos is None:
            init_pos = [0, 0, 0, 0, 0, 0]
        self.Logger.debug("Add model %s in %s." % (model_name, init_pos,))
        return self.addModelXml(model_name, model_xml, init_pos, reference_frame)

    def delModel(self, model_name):
        """
        :param model_name: model name in gazebo you want to delete.
        :return: if delete success
        """
        self.Logger.debug("Del model %s ." % (model_name,))
        res = SetDeleteModelReq(model_name)
        return res.success

    def ControlModelStatic(self, model_name, BOOL):
        """

        :param model_name: model name in gazebo you want to static
        :param BOOL: control model static or not
        :return: if control static success
        """
        self.Logger.debug("%s is %s static" % (model_name, BOOL))
        res = SetBoolReq(BOOL, '/' + model_name + '/control_static')
        return res.success
