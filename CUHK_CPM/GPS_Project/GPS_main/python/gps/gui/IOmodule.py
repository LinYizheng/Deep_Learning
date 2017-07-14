#!/usr/bin/env python

# ROS libs
import rospy

# ROS rosservice
from std_srvs.srv import SetBool

class Server():
    def __init__(self, rosservice):
        self.Server = rospy.Service(rosservice, SetBool, self.RespondFunction)
        self.Y = False

    def RespondFunction(self, req_x):
        self.Y = req_x.data
        return True, "sucess rosservice call, Y = " + str(self.Y)

    def Get_output(self):
        return self.Y