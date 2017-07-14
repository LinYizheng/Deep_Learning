#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '6/1/2017'
__copyright__ = "Copyright 2016, PI"
__all__ = [
    'RobotControlErr',
    'RobotControl',
]
import os
import sys
__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
import com_interface as Com
import logging
import numpy as np

class RobotControlErr(Exception):
    pass

class RobotControl(object):
    def __init__(self):
        object.__init__(self)
        self.__Logger = logging.getLogger(self.__class__.__name__)
        self.__ShmBuf = Com.link_port("shm_VS.bin")
        self.setSpeed([20, 20, 20])
        self.setAccel([20, 20, 20, 20, 20, 20])
        if self.__ShmBuf:
            self.__Logger.info("Success to Connect!!")
        else:
            self.__Logger.warn("Failed to Connect!!")

    @property
    def Logger(self):
        """
        :return: pointer to logger
        """
        return self.__Logger

    def pubPosition(self, Mode, Position, waite=True):
        """

        :param Mode:  'Go' or 'Move'
        :param Position: if sixAxis: [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
        :param waite: if waite robot reach the pos
        :return: if fourAxis: [x_mm, y_mm, z_mm, u_deg, 0, 0]
        """
        assert isinstance(Mode, (str, ))
        assert isinstance(Position, (np.ndarray, list, tuple))
        assert len(Position) == 6, 'Position has 6 values'
        self.Logger.debug('pubPosition: %r' % (Position,))
        Com.command_to_robot(self.__ShmBuf, str.upper(Mode), Position, wait=waite)

    def getPosition(self):
        """

        :return: Feedback position of robot　6_axis: [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
                                            4_axis:[x_mm, y_mm, z_mm, u_deg]
        """
        FbPosition = Com.get_status(self.__ShmBuf, "WHERE")
        self.Logger.debug('getPosition: %r' % (FbPosition,))
        return FbPosition

    def pubAngle(self, Angle):
        """

        :param Angle: if sixAxis: [J1_deg, J2_deg, J3_deg, J4_deg, J5_deg, J6_deg]
                      if fourAxis: [J1_deg, J2_deg, J3_deg, J4_deg, 0, 0]
        :return: None
        """
        assert isinstance(Angle, (np.ndarray, list, tuple))
        assert len(Angle) == 6, 'Angle has 6 values'
        self.Logger.debug('pubAngle: %r' % (Angle,))
        Com.command_to_robot(self.__ShmBuf, "GO_DEG", Angle)

    def getAngle(self):
        """

        :return: Feedback position of robot 6_axis: [J1_deg, J2_deg, J3_deg, J4_deg, J5_deg, J6_deg]
                                            4_axis: [J1_deg, J2_deg, J3_deg, J4_deg]
        """
        FbAngle = Com.get_status(self.__ShmBuf, "WHERE_DEG")
        self.Logger.debug('getAngle: %r' % (FbAngle,))
        return FbAngle

    def setSpeed(self, Speed):
        """
        :param Speed: [speed_percent, Depart_speed_percent, approach_speed_percent]
        :return: None
        """
        assert isinstance(Speed, (np.ndarray, list, tuple))
        assert len(Speed) == 3, 'Speed has 3 values'
        self.Logger.debug('setSpeed: %r' % (Speed,))
        Com.command_to_robot(self.__ShmBuf, "SPEED", Speed)

    def getSpeed(self):
        """

        :return: Current Speed of robot 【speed_percent, Depart_speed_percent, approach_speed_percent]
        """
        FbSpeed = Com.get_status(self.__ShmBuf, "SPEED")
        self.Logger.debug('getSpeed: %r' % (FbSpeed,))
        return FbSpeed

    def setAccel(self, Accel):
        """

        :param Accel: [accel_percent, decel_percent, depart_accel, depart_decel, approach_accel. approach_decel]
        :return:
        """
        assert isinstance(Accel, (np.ndarray, list, tuple))
        assert len(Accel) == 6, 'Accel has 6 values'
        self.Logger.debug('setAccel: %r' % (Accel,))
        Com.command_to_robot(self.__ShmBuf, "ACCEL", Accel)

    def getAccel(self):
        """

        :return: Current Accel of robot　[accel_percent, decel_percent, depart_accel, depart_decel, approach_accel. approach_decel]
        """
        FbAccel = Com.get_status(self.__ShmBuf, "ACCEL")
        self.Logger.debug('getSpeed: %r' % (FbAccel,))
        return FbAccel

    def setElbow(self, Elbow):
        """

        :param Elbow: 'ABOVE', 'BELOW',
        :return: None
        """
        assert isinstance(Elbow, (str,))
        if str.upper(Elbow) == "ABOVE":
            self.Logger.debug('setElbow: ABOVE')
            Com.command_to_robot(self.__ShmBuf, "ELBOW", 0)
        elif str.upper(Elbow) == "BELOW":
            self.Logger.debug('setElbow: BELOW')
            Com.command_to_robot(self.__ShmBuf, "ELBOW", 1)
        else:
            self.Logger.warn("setElbow function: 'ABOVE' or 'BELOW' be required")

    def setHand(self, Hand):
        """

        :param Hand: 'LEFT', 'RIGHT', 'AUTO'
        :return: None
        """
        assert isinstance(Hand, (str,))
        if str.upper(Hand) == "LEFT":
            self.Logger.debug('setHand: LEFT.')
            Com.command_to_robot(self.__ShmBuf, "HAND", 0)
        elif str.upper(Hand) == "RIGHT":
            self.Logger.debug('setHand: RIGHT.')
            Com.command_to_robot(self.__ShmBuf, "HAND", 1)
        elif str.upper(Hand) == "AUTO":
            self.Logger.debug('setHand: AUTO.')
            Com.command_to_robot(self.__ShmBuf, "HAND", 2)
        else:
            self.Logger.warn("setElbow function: 'LEFT', 'RIGHT' or 'AUTO' be required")

    def isTargetOk(self, Pos):
        """

        :param Pos: [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
        :return: result of IsTargetOk,0 for ok, non-zero for not ok.
        """
        self.Logger.debug('checkTargetOk: %r' % (Pos,))
        Result = Com.get_status(self.__ShmBuf, "TARGET_OK", Pos)
        return Result


if __name__ == '__main__':
    import numpy as np
    import time
    Robot = RobotControl()
    print Robot.getSpeed()[:]
    print Robot.getAccel()[:]
    Robot.pubPosition('go', np.array([350, 100, 100, 0, 0, 10]))
    time.sleep(1.0)
    Robot.pubAngle(np.array([50, 0, 0, 0, 0, 0]))
    time.sleep(1.0)
