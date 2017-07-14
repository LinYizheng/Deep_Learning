#!/usr/bin/env python
# numpy
import numpy as np

# ROS libs
import rospy
# ROS messages
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# math
import math
# time
import time
# ROS server
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


class Communication:
    def __init__(self):
        """Initialize ros node, ros publisher, ros subscriber"""
        rospy.init_node('Communication', anonymous=True)  # node init
        # topic where we publish
        self.Angle_pub = rospy.Publisher("/mybot/Joint_Angle_Controller", Float64MultiArray, queue_size=10)
        self.Position_pub = rospy.Publisher("/mybot/Joint_Position_Controller", Float64MultiArray, queue_size=10)
        self.Control_Method_pub = rospy.Publisher("/mybot/Position_ControlMethod_Controller", Float64MultiArray,
                                                  queue_size=10)

        # self.left_gripper_pub = rospy.Publisher("/pi/left_finger_controller/command", Float64, queue_size=10)
        # self.right_gripper_pub = rospy.Publisher("/pi/right_finger_controller/command", Float64, queue_size=10)
        # subscribed topic
        self.Joint_states_angle_sub = rospy.Subscriber("/mybot/Joint_AngleDeg_State", Float64MultiArray,
                                                       self.Angelecallback)
        self.Joint_states_position_sub = rospy.Subscriber("/mybot/Joint_PositionMmDeg_State", Float64MultiArray,
                                                          self.Positioncallback)
        self.Joint_states_velocity_sub = rospy.Subscriber("/mybot/Joint_VelocityDegPerS_State", Float64MultiArray,
                                                          self.Velocitycallback)
        self.Joint_states_ControlMethod_sub = rospy.Subscriber("/mybot/Position_ControlMethod_State", Float64MultiArray,
                                                               self.ControlMethodcallback)

        self.image_sub = rospy.Subscriber('/depth/link/raw/image', Image, self.ImageCallback)
        # image_sub = message_filters.Subscriber('/depth/raw/image', Image)
        # depth_sub = message_filters.Subscriber('/depth/depth/image', Image)

        # ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
        # ts.registerCallback(self.ImageCallback)

        self.OriAngle = np.zeros(4)

        self.FeedbackAngle = np.zeros(6)
        self.FeedbackPosition = np.zeros(6)
        self.FeedbackVelocity = np.zeros(6)
        self.ModeState = 0
        self.ElbowState = 0
        self.percentage = 50
        self.IkState = 0
        self.RGBData = np.zeros(480 * 480 * 3)
        # self.DepthData = np.zeros(480 * 480)
        self.PubAngle = Float64MultiArray()
        self.PubPosition = Float64MultiArray()
        self.PubControlMethod = Float64MultiArray()
        self.PubControlMethod.data = np.zeros(3)
        self.left_finger = Float64()
        self.right_finger = Float64()
        self.bridge = CvBridge()
        self.WriteFlag = False
        time.sleep(2)  # for ros to connect ros topic success

    def ImageCallback(self, image):
        try:
            self.RGBData = self.bridge.imgmsg_to_cv2(image, "bgr8")
            # self.DepthData = self.bridge.imgmsg_to_cv2(depth, "passthrough")
            self.WriteFlag = True
        except CvBridgeError as e:
            print(e)

    def Angelecallback(self, joint_state_angle):
        """
        feedback angle 0-5: angle
        :param joint_state_angle:
        :return:
        """
        self.FeedbackAngle = np.array(joint_state_angle.data[:], dtype=np.float64)

    def Velocitycallback(self, joint_state_velocity):
        """
        feedback velocity 0-5: velocity
        :param joint_state_velocity:
        :return:
        """
        self.FeedbackVelocity = np.array(joint_state_velocity.data[:], dtype=np.float64)

    def ControlMethodcallback(self, joint_state_control_method):
        """
        :param joint_state_control_method:
        :return:
        """
        self.ModeState = joint_state_control_method.data[0]
        self.ElbowState = joint_state_control_method.data[1]
        self.percentage = joint_state_control_method.data[2]
        self.IkState = joint_state_control_method.data[3]

    def Positioncallback(self, joint_state_position):
        """
        feedback position 0-5: position 
        :param joint_state_position:
        :return:
        """
        self.FeedbackPosition = np.array(joint_state_position.data[:], dtype=np.float64)

    def Ola2Ori(self, Y, P, R):
        """ Ola convert to Ori Angel"""
        self.OriAngle[0] = math.cos(Y / 2) * math.cos(P / 2) * math.cos(R / 2) + math.sin(Y / 2) * math.sin(
            P / 2) * math.sin(R / 2)
        self.OriAngle[1] = math.sin(Y / 2) * math.cos(P / 2) * math.cos(R / 2) - math.cos(Y / 2) * math.sin(
            P / 2) * math.sin(R / 2)
        self.OriAngle[2] = math.cos(Y / 2) * math.sin(P / 2) * math.cos(R / 2) + math.sin(Y / 2) * math.cos(
            P / 2) * math.sin(R / 2)
        self.OriAngle[3] = math.cos(Y / 2) * math.cos(P / 2) * math.sin(R / 2) - math.sin(Y / 2) * math.sin(
            P / 2) * math.cos(R / 2)

    def Pub_Position(self, joint_pos_mm_deg):
        """
        publish position np.array([X,Y,Z,U,V,W]) [(x,y,z)Mm and (u,v,w)Degree,period is control update rate]
        :param joint_pos_mm_deg:
        :return:
        """
        self.PubPosition.data = joint_pos_mm_deg
        self.Position_pub.publish(self.PubPosition)

    def Pub_Angle(self, joint_angle_deg):
        """
        publish angle np.array([J1 - J6]) (degree)
        :param joint_angle_deg:
        :return:
        """
        self.PubAngle.data = joint_angle_deg
        self.Angle_pub.publish(self.PubAngle)

    def Pub_control_method(self, control_method):
        """
        :param control_method: [MODE,ELBOW,PERCENT]
        :return:
        """
        self.PubControlMethod.data = control_method
        self.Control_Method_pub.publish(self.PubControlMethod)

    def Select_Mode_Go(self):
        self.PubControlMethod.data[0] = 0  # GO
        self.PubControlMethod.data[1] = self.ElbowState
        self.PubControlMethod.data[2] = self.percentage
        self.Control_Method_pub.publish(self.PubControlMethod)

    def Select_Mode_Move(self):
        self.PubControlMethod.data[0] = 1  # Move
        self.PubControlMethod.data[1] = self.ElbowState
        self.PubControlMethod.data[2] = self.percentage
        self.Control_Method_pub.publish(self.PubControlMethod)

    def Select_Elbow_Above(self):
        self.PubControlMethod.data[0] = self.ModeState
        self.PubControlMethod.data[1] = 0  # Above
        self.PubControlMethod.data[2] = self.percentage
        self.Control_Method_pub.publish(self.PubControlMethod)

    def Select_Elbow_Below(self):
        self.PubControlMethod.data[0] = self.ModeState
        self.PubControlMethod.data[1] = 1  # Below
        self.PubControlMethod.data[2] = self.percentage
        self.Control_Method_pub.publish(self.PubControlMethod)

    def Set_Send_percentage(self, percentage):
        """
        :param percentage:
        :return:
        """
        self.PubControlMethod.data[0] = self.ModeState
        self.PubControlMethod.data[1] = self.ElbowState
        self.PubControlMethod.data[2] = percentage  # send period
        self.Control_Method_pub.publish(self.PubControlMethod)

    def Get_angle(self):
        """
        get feedback ange
        :return: np.array[J1 - J6] (Degree)
        """
        return self.FeedbackAngle

    def Get_velocity(self):
        """
        get feedback ange
        :return: np.array[J1 - J6] (Degree)
        """
        return self.FeedbackVelocity

    def Get_position(self):
        """
        get feedback position
        :return: np.array[X,Y,Z,U,V,W] [Mm,Degree]
        """
        return self.FeedbackPosition

    def Get_IkSate(self):
        """
        if == 0 fail IkSolution
        if == 1 success IkSolution
        :return:
        """
        return self.IkState

    def Get_ModeState(self):
        """
        if == 0  GO
        if == 1 Move
        :return:
        """
        return self.ModeState

    def Get_ElbowState(self):
        """
        if == 0  ABOVE
        if == 1  BELOW
        :return:
        """
        return self.ElbowState

    def Get_percentage(self):
        """
        :return:  speed
        """
        return self.percentage

    def Get_image_RGB(self):
        """
        :return: image of rgb
        """
        self.WriteFlag = False
        while not self.WriteFlag:
            pass
        return self.RGBData

    # def Get_image_Depth(self):
    #     """
    #     :return image of depth
    #     :return:
    #     """
    #     return self.DepthData

    # def PubGripperPos(self, left_finger_pos, right_finger_pos):
    #     """
    #     control gripper position
    #     """
    #     self.left_finger = left_finger_pos
    #     self.right_finger = right_finger_pos
    #     self.left_gripper_pub.publish(self.left_finger)
    #     self.right_gripper_pub.publish(self.right_finger)
    #
    # def ColseGripper(self):
    #     rospy.wait_for_service('/control_X11_State')
    #     try:
    #         Set_Bool = rospy.ServiceProxy('/control_X1_State', SetBool)
    #         resp1 = Set_Bool(True)
    #         return resp1.success
    #     except rospy.ServiceException, e:
    #         print "Service call failed: %s"%e
    #
    # def OpenGripper(self):
    #     rospy.wait_for_service('/control_X11_State')
    #     try:
    #         Set_Bool = rospy.ServiceProxy('/control_X1_State', SetBool)
    #         resp1 = Set_Bool(False)
    #         return resp1.success
    #     except rospy.ServiceException, e:
    #         print "Service call failed: %s"%e


    def Set_Object_Pos(self, model_name, position):
        self.Ola2Ori(position[3], position[4], position[5])
        PosState = ModelState()
        PosState.model_name = model_name
        PosState.pose.position.x = position[0]
        PosState.pose.position.y = position[1]
        PosState.pose.position.z = position[2]
        PosState.pose.orientation.x = self.OriAngle[1]
        PosState.pose.orientation.y = self.OriAngle[2]
        PosState.pose.orientation.z = self.OriAngle[3]
        PosState.pose.orientation.w = self.OriAngle[0]
        PosState.twist.linear.x = 0.0
        PosState.twist.linear.y = 0.0
        PosState.twist.linear.z = 0.0
        PosState.twist.angular.x = 0.0
        PosState.twist.angular.y = 0.0
        PosState.twist.angular.z = 0.0
        PosState.reference_frame = "world"
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp1 = model_state(PosState)
            return resp1.success
        except rospy.ServiceException as e:
            print "Service call failed: %s" % e
