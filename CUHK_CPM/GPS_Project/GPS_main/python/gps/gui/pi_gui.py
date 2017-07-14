#!/usr/bin/env python
import os
import wx
import numpy as np
import time
import rospy
from std_srvs.srv import SetBool
from gps.agent.pi.pi_robot_API import Communication
from IOmodule import Server
import logging

LOGGER = logging.getLogger('base')

class ControlPanel(wx.Frame):
    def __init__(self, parent, npzFileName):
        self.npzFileName = npzFileName
        wx.Frame.__init__(self, parent, size=(600, 500))
        ######################set Graduation in solider #######
        self.Graduation = 0.1
        self.wei = int(0.1 * 10)
        ##################### write value ###############
        self.joint_pos_deg = np.zeros(6)
        self.xyzuvw = np.zeros(6)
        self.ControlMethod = np.zeros(3)
        self.dispy_joint_pos = np.zeros(6)
        self.dispy_coord = np.zeros(6)
        self.Ik_state = 0
        self.ModeState = 0
        self.ElbowState = 0
        self.percentage = 0
        ###############################
        self.pnl = wx.Panel(self)
        self.Communication = Communication()
        time.sleep(0.2)
        self.dispy_joint_pos = self.Communication.Get_angle()
        self.dispy_coord = self.Communication.Get_position()
        self.dispy_joint_vel = self.Communication.Get_velocity()
        self.percentage = self.Communication.Get_percentage()
        ################################plc i/o ################################## 
        ########################creat i/o server ##############################
        self.ServerX1 = Server("/control_X1_State")
        self.ServerX2 = Server("/control_X2_State")
        self.ServerX3 = Server("/control_X3_State")
        self.ServerX4 = Server("/control_X4_State")
        self.ServerX5 = Server("/control_X5_State")
        self.ServerX6 = Server("/control_X6_State")
        self.ServerX7 = Server("/control_X7_State")
        self.ServerX8 = Server("/control_X8_State")
        self.ServerX9 = Server("/control_X9_State")
        self.ServerX10 = Server("/control_X10_State")
        self.ServerX11 = Server("/control_X11_State")
        self.ServerX12 = Server("/control_X12_State") 
        ####################call server ##########################    
        wx.StaticBox(self.pnl, label="I/O", pos=(495, 0), size=(100, 395))
        self.X1 = CheckBox(self.pnl, "X1", 500, 15, '/control_X1_State') 
        self.X2 = CheckBox(self.pnl, "X2", 500, 45, '/control_X2_State')
        self.X3 = CheckBox(self.pnl, "X3", 500, 75, '/control_X3_State')
        self.X4 = CheckBox(self.pnl, "X4", 500, 105, '/control_X4_State')
        self.X5 = CheckBox(self.pnl, "X5", 500, 135, '/control_X5_State')
        self.X6 = CheckBox(self.pnl, "X6", 500, 165, '/control_X6_State')
        self.X7 = CheckBox(self.pnl, "X7", 500, 195, '/control_X7_State')
        self.X8 = CheckBox(self.pnl, "X8", 500, 225, '/control_X8_State')
        self.X9 = CheckBox(self.pnl, "X9", 500, 255, '/control_X9_State')
        self.X10 = CheckBox(self.pnl, "X10", 500, 285, '/control_X10_State')
        self.X11 = CheckBox(self.pnl, "X11", 500, 315, '/control_X11_State')
        self.X12 = CheckBox(self.pnl, "X12", 500, 345, '/control_X12_State')       
        ####joint position solider #########
        wx.StaticBox(self.pnl, label="position control(deg)", pos=(5, 0), size=(335, 150))
        ############################J1#########################
        wx.StaticText(self.pnl, label="J1", pos=(10 + 10, 5 + 10))
        self.J1 = wx.Slider(self.pnl, id=wx.NewId(), value=round(self.dispy_joint_pos[0], self.wei) / self.Graduation,
                            minValue=-149.0 / self.Graduation,
                            maxValue=149.0 / self.Graduation, pos=(30 + 10, 10), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.J1.Bind(wx.EVT_SCROLL, self.OnJ1SliderScroll)
        self.J1_val = wx.StaticText(self.pnl, label=str(round(self.dispy_joint_pos[0], self.wei)),
                                    pos=(280 + 20, 5 + 10))
        ############################J2#########################
        wx.StaticText(self.pnl, label="J2", pos=(10 + 10, 5 + 30))
        self.J2 = wx.Slider(self.pnl, id=wx.NewId(), value=round(self.dispy_joint_pos[1], self.wei) / self.Graduation,
                            minValue=-51.6 / self.Graduation,
                            maxValue=97.4 / self.Graduation, pos=(30 + 10, 30), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.J2.Bind(wx.EVT_SCROLL, self.OnJ2SliderScroll)
        self.J2_val = wx.StaticText(self.pnl, label=str(round(self.dispy_joint_pos[1], self.wei)),
                                    pos=(280 + 20, 5 + 30))
        ############################J2#########################
        wx.StaticText(self.pnl, label="J3", pos=(10 + 10, 5 + 50))
        self.J3 = wx.Slider(self.pnl, id=wx.NewId(), value=round(self.dispy_joint_pos[2], self.wei) / self.Graduation,
                            minValue=-206.3 / self.Graduation,
                            maxValue=34.4 / self.Graduation, pos=(30 + 10, 50), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.J3.Bind(wx.EVT_SCROLL, self.OnJ3SliderScroll)
        self.J3_val = wx.StaticText(self.pnl, label=str(round(self.dispy_joint_pos[2], self.wei)),
                                    pos=(280 + 20, 5 + 50))
        ############################J1#########################
        wx.StaticText(self.pnl, label="J4", pos=(10 + 10, 5 + 70))
        self.J4 = wx.Slider(self.pnl, id=wx.NewId(), value=round(self.dispy_joint_pos[3], self.wei) / self.Graduation,
                            minValue=-180 / self.Graduation,
                            maxValue=180 / self.Graduation, pos=(30 + 10, 70), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.J4.Bind(wx.EVT_SCROLL, self.OnJ4SliderScroll)
        self.J4_val = wx.StaticText(self.pnl, label=str(round(self.dispy_joint_pos[3], self.wei)),
                                    pos=(280 + 20, 5 + 70))
        ############################J2#########################
        wx.StaticText(self.pnl, label="J5", pos=(10 + 10, 5 + 90))
        self.J5 = wx.Slider(self.pnl, id=wx.NewId(), value=round(self.dispy_joint_pos[4], self.wei) / self.Graduation,
                            minValue=-107.2 / self.Graduation,
                            maxValue=107.2 / self.Graduation, pos=(30 + 10, 90), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.J5.Bind(wx.EVT_SCROLL, self.OnJ5SliderScroll)
        self.J5_val = wx.StaticText(self.pnl, label=str(round(self.dispy_joint_pos[4], self.wei)),
                                    pos=(280 + 20, 5 + 90))
        ############################J2#########################
        wx.StaticText(self.pnl, label="J6", pos=(10 + 10, 5 + 110))
        self.J6 = wx.Slider(self.pnl, id=wx.NewId(), value=round(self.dispy_joint_pos[5], self.wei) / self.Graduation,
                            minValue=-360 / self.Graduation,
                            maxValue=360 / self.Graduation, pos=(30 + 10, 110), size=(250, -1), style=wx.SL_HORIZONTAL)
        self.J6.Bind(wx.EVT_SCROLL, self.OnJ6SliderScroll)
        self.J6_val = wx.StaticText(self.pnl, label=str(round(self.dispy_joint_pos[5], self.wei)),
                                    pos=(280 + 20, 5 + 110))

        ############ coordnate TextCtrl #####################
        wx.StaticBox(self.pnl, label="coordnate control", pos=(340, 10), size=(150, 300))
        self.CoordSetValueX = coordnate(self.pnl, "X:", round(self.dispy_coord[0], 2), 100, -1, 380, 30)
        self.CoordSetValueY = coordnate(self.pnl, "Y:", round(self.dispy_coord[1], 2), 100, -1, 380, 60)
        self.CoordSetValueZ = coordnate(self.pnl, "Z:", round(self.dispy_coord[2], 2), 100, -1, 380, 90)
        self.CoordSetValueU = coordnate(self.pnl, "U:", round(self.dispy_coord[3], 2), 100, -1, 380, 120)
        self.CoordSetValueV = coordnate(self.pnl, "V:", round(self.dispy_coord[4], 2), 100, -1, 380, 150)
        self.CoordSetValueW = coordnate(self.pnl, "W:", round(self.dispy_coord[5], 2), 100, -1, 380, 180)
        ###################coordnate display #######################
        wx.StaticBox(self.pnl, label="coordnate display", pos=(5, 150), size=(335, 80))
        self.displayX = display_pos(self.pnl, "X:", 0, 40, 170)
        self.displayY = display_pos(self.pnl, "Y:", 0, 150, 170)
        self.displayZ = display_pos(self.pnl, "Z:", 0, 260, 170)
        self.displayU = display_pos(self.pnl, "U:", 0, 40, 200)
        self.displayV = display_pos(self.pnl, "V:", 0, 150, 200)
        self.displayW = display_pos(self.pnl, "W:", 0, 260, 200)
        ###################position display #######################
        wx.StaticBox(self.pnl, label="position display(deg)", pos=(5, 230), size=(335, 80))
        self.displayJ1Q = display_pos(self.pnl, "J1:", 0, 40, 250)
        self.displayJ2Q = display_pos(self.pnl, "J2:", 0, 150, 250)
        self.displayJ3Q = display_pos(self.pnl, "J3:", 0, 260, 250)
        self.displayJ4Q = display_pos(self.pnl, "J4:", 0, 40, 280)
        self.displayJ5Q = display_pos(self.pnl, "J5:", 0, 150, 280)
        self.displayJ6Q = display_pos(self.pnl, "J6:", 0, 260, 280)
        ###################position display #######################
        wx.StaticBox(self.pnl, label="velocity display(deg/s)", pos=(5, 310), size=(335, 80))
        self.displayJ1Qdot = display_pos(self.pnl, "J1:", 0, 40, 330)
        self.displayJ2Qdot = display_pos(self.pnl, "J2:", 0, 150, 330)
        self.displayJ3Qdot = display_pos(self.pnl, "J3:", 0, 260, 330)
        self.displayJ4Qdot = display_pos(self.pnl, "J4:", 0, 40, 360)
        self.displayJ5Qdot = display_pos(self.pnl, "J5:", 0, 150, 360)
        self.displayJ6Qdot = display_pos(self.pnl, "J6:", 0, 260, 360)
        ###############GO button#########################
        self.GoButton = wx.Button(self.pnl, -1, u'Run', size=(70, 40), pos=(415, 345))
        self.GoButton.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        self.Bind(wx.EVT_BUTTON, self.OnRun, self.GoButton)
        ####################HOME################
        self.HomeButton = wx.Button(self.pnl, -1, u'Home', size=(70, 40), pos=(345, 345))
        self.HomeButton.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        self.Bind(wx.EVT_BUTTON, self.onHome, self.HomeButton)
        ################display IK result ############################ 
        wx.StaticBox(self.pnl, label="IK", pos=(350, 210), size=(50, 50))
        self.Dis_Ik_result = wx.StaticText(self.pnl, label="", pos=(355, 230))
        #########################choice mode ################
        wx.StaticText(self.pnl, label="Mode:", pos=(400, 210))
        self.Dis_Mode = wx.StaticText(self.pnl, label="Go", pos=(450, 210))
        self.Dis_Mode.SetForegroundColour("red")
        self.mode_type = ["Go", "Move"]
        self.mode_choice = wx.Choice(self.pnl, -1, size=(80, 30), pos=(400, 225), choices=self.mode_type)
        self.mode_choice.Bind(wx.EVT_CHOICE, self.OnModeChoice)
        #########################ELBOW_ mode ################
        wx.StaticText(self.pnl, label="Elbow:", pos=(400, 255))
        self.Dis_ELBOW = wx.StaticText(self.pnl, label="ABO", pos=(450, 255))
        self.Dis_ELBOW.SetForegroundColour("red")
        self.ELBOW_type = ["ABOVE", "BELOW"]
        self.ELBOW_choice = wx.Choice(self.pnl, -1, size=(80, 30), pos=(400, 270), choices=self.ELBOW_type)
        self.ELBOW_choice.Bind(wx.EVT_CHOICE, self.OnElbowChoice)
        ########################################################################################
        wx.StaticBox(self.pnl, label="speed", pos=(350, 260), size=(50, 40))
        wx.StaticBox(self.pnl, label="speed(%)", pos=(345, 305), size=(140, 40))
        self.Dis_speed = wx.StaticText(self.pnl, label=str(self.percentage), pos=(365, 280))
        self.Dis_speed.SetForegroundColour("red")
        self.period = wx.Slider(self.pnl, id=wx.NewId(), value=self.percentage,
                                minValue=1,
                                maxValue=100, pos=(335 + 10, 315), size=(140, -1), style=wx.SL_HORIZONTAL)
        self.period.Bind(wx.EVT_SCROLL, self.OnperiodSliderScroll)
        self.period_val = wx.StaticText(self.pnl, label=str(self.percentage), pos=(420, 305))
        time.sleep(0.1)
        self.Communication.Select_Mode_Go()
        time.sleep(0.1)
        self.Communication.Select_Elbow_Above()
        ################ gps button#######################################
        wx.StaticBox(self.pnl, label="gps", pos=(5, 390), size=(235, 90))
        #######################set init pos#################################3
        self.SetIntPtsButton = wx.Button(self.pnl, -1, u'SetInitPts', size=(90, 30), pos=(5, 410))
        self.SetIntPtsButton.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        self.Bind(wx.EVT_BUTTON, self.OnSetIntPts, self.SetIntPtsButton)
        #######################set target pos#############################
        self.SetTgtPtsButton = wx.Button(self.pnl, -1, u'SetTgtPts', size=(90, 30), pos=(5, 440))
        self.SetTgtPtsButton.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        self.Bind(wx.EVT_BUTTON, self.OnSetTgtPts, self.SetTgtPtsButton)
        #######################mov to init pos#################################3
        self.MovIntPtsButton = wx.Button(self.pnl, -1, u'MovInitPts', size=(90, 30), pos=(95, 410))
        self.MovIntPtsButton.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        self.Bind(wx.EVT_BUTTON, self.OnMovIntPts, self.MovIntPtsButton)
        #######################mov to target pos#############################
        self.MovTgtPtsButton = wx.Button(self.pnl, -1, u'MovTgtPts', size=(90, 30), pos=(95, 440))
        self.MovTgtPtsButton.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        self.Bind(wx.EVT_BUTTON, self.OnMovTgtPts, self.MovTgtPtsButton)
        ###########################key text########################
        # wx.StaticText(self.pnl, label="Num", pos=(115, 310))
        self.Cordition = wx.TextCtrl(self.pnl, -1, "0", size=(50, 50), style=wx.TE_CENTRE, pos=(185, 420))
        self.Cordition.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        #########################disply save data state #####################
        self.txt = wx.StaticBox(self.pnl, label="write data display", pos=(245, 390), size=(250, 90))
        self.Dis_npz_state = wx.StaticText(self.pnl, label="", pos=(230, 410))
        ################# set refresh timer 10MS####################
        self.timer = wx.Timer(self, wx.NewId())
        self.Bind(wx.EVT_TIMER, self.DataUpdate)
        self.timer.Start(10)
        self.Centre()
        self.Show(True)

    def save_to_npz(self, filename, key, value):
        """
        Save a (key,value) pair to a npz dictionary.
        Args:
            filename: The file containing the npz dictionary.
            key: The key (string).
            value: The value (numpy array).
        """
        tmp = {}
        if os.path.exists(filename):
            with np.load(filename) as f:
                tmp = dict(f)
        tmp[key] = value
        np.savez(filename, **tmp)

    def read_from_npz(self, filename, key):
        tmp = {}
        if os.path.exists(filename):
            with np.load(filename) as f:
                tmp = dict(f)
        if key in tmp:
            self.value = tmp[key]
            return True
        else:
            return False

    def OnMovIntPts(self, event):
        self.key = self.Cordition.GetValue()
        if len(self.key) == 1:
            if len(self.key) == 1 and ord(self.key) >= 48 and ord(self.key) <= 57 and len(self.key) == 1:
                if self.read_from_npz(self.npzFileName, "initial" + self.key):
                    self.Communication.Pub_Angle(self.value)
                    # shm.shm_write(self.shm_buf, 1, self.shm_size, 1, 1, 1)  # go_flag
                    # shm.shm_write(self.shm_buf, 1, self.shm_size, 0, 1, 1)  ##chose coord mode
                    # shm.shm_write(self.shm_buf, shm.float2uchar(self.value, 24), self.shm_size, 27, 24, 1)
                    # shm.shm_write(self.shm_buf, 0, self.shm_size, 1, 1, 1)  ##chose joint mode
                    # shm.shm_write(self.shm_buf, shm.float2uchar(self.value, 24), self.shm_size, 2, 24, 1)
                    # shm.shm_write(self.shm_buf, 2, self.shm_size, 1, 1, 1)  ##chose tau mode
                    # shm.shm_write(self.shm_buf, shm.float2uchar(self.value, 24), self.shm_size, 153, 24, 1)
                    self.Dis_npz_state.SetLabel(
                        "             Move to Initial Position!    " + self.key + " : = \n       [" + str(
                            self.value[0]) + "     " + str(
                            self.value[1]) \
                        + "\n       " + str(self.value[2]) + "       " + str(self.value[3]) + "\n       " + str(
                            self.value[4]) + \
                        "         " + str(self.value[5]) + "]")
                    self.Dis_npz_state.SetForegroundColour("green")
                    ###############update solider value#############
                    self.J1.SetValue(round(self.value[0], self.wei) / self.Graduation)
                    self.J1_val.SetLabel(str(round(self.value[0], self.wei)))
                    self.joint_pos_deg[0] = self.value[0]
                    self.J2.SetValue(round(self.value[1], self.wei) / self.Graduation)
                    self.J2_val.SetLabel(str(round(self.value[1], self.wei)))
                    self.joint_pos_deg[1] = self.value[1]
                    self.J3.SetValue(round(self.value[2], self.wei) / self.Graduation)
                    self.J3_val.SetLabel(str(round(self.value[2], self.wei)))
                    self.joint_pos_deg[2] = self.value[2]
                    self.J4.SetValue(round(self.value[3], self.wei) / self.Graduation)
                    self.J4_val.SetLabel(str(round(self.value[3], self.wei)))
                    self.joint_pos_deg[3] = self.value[3]
                    self.J5.SetValue(round(self.value[4], self.wei) / self.Graduation)
                    self.J5_val.SetLabel(str(round(self.value[4], self.wei)))
                    self.joint_pos_deg[4] = self.value[4]
                    self.J6.SetValue(round(self.value[5], self.wei) / self.Graduation)
                    self.J6_val.SetLabel(str(round(self.value[5], self.wei)))
                    self.joint_pos_deg[5] = self.value[5]
                else:
                    self.Dis_npz_state.SetLabel("             This key is not exist!")
                    self.Dis_npz_state.SetForegroundColour("red")

            else:
                self.Dis_npz_state.SetLabel("           Enter num error!")
                self.Dis_npz_state.SetForegroundColour("red")
        else:
            self.Dis_npz_state.SetLabel("           Enter num error!")
            self.Dis_npz_state.SetForegroundColour("red")

    def OnMovTgtPts(self, event):
        self.key = self.Cordition.GetValue()
        if len(self.key) == 1:
            if len(self.key) == 1 and ord(self.key) >= 48 and ord(self.key) <= 57 and len(self.key) == 1:
                if self.read_from_npz(self.npzFileName, "target" + self.key):
                    self.Communication.Pub_Angle(self.value)
                    # shm.shm_write(self.shm_buf, 1, self.shm_size, 1, 1, 1)  # go_flag
                    # shm.shm_write(self.shm_buf, 1, self.shm_size, 0, 1, 1)  ##chose coord mode
                    # shm.shm_write(self.shm_buf, shm.float2uchar(self.value, 24), self.shm_size, 27, 24, 1)
                    # shm.shm_write(self.shm_buf, 0, self.shm_size, 1, 1, 1)  ##chose joint mode
                    # shm.shm_write(self.shm_buf, shm.float2uchar(self.value, 24), self.shm_size, 2, 24, 1)
                    # shm.shm_write(self.shm_buf, 2, self.shm_size, 1, 1, 1)  ##chose tau mode
                    # shm.shm_write(self.shm_buf, shm.float2uchar(self.value, 24), self.shm_size, 153, 24, 1)
                    self.Dis_npz_state.SetLabel(
                        "             Move to Target Position!    " + self.key + " : = \n       [" + str(
                            self.value[0]) + "     " + str(
                            self.value[1]) \
                        + "\n       " + str(self.value[2]) + "       " + str(self.value[3]) + "\n       " + str(
                            self.value[4]) + \
                        "         " + str(self.value[5]) + "]")
                    self.Dis_npz_state.SetForegroundColour("blue")
                    ###############update solider value#############
                    self.J1.SetValue(round(self.value[0], self.wei) / self.Graduation)
                    self.J1_val.SetLabel(str(round(self.value[0], self.wei)))
                    self.joint_pos_deg[0] = self.value[0]
                    self.J2.SetValue(round(self.value[1], self.wei) / self.Graduation)
                    self.J2_val.SetLabel(str(round(self.value[1], self.wei)))
                    self.joint_pos_deg[1] = self.value[1]
                    self.J3.SetValue(round(self.value[2], self.wei) / self.Graduation)
                    self.J3_val.SetLabel(str(round(self.value[2], self.wei)))
                    self.joint_pos_deg[2] = self.value[2]
                    self.J4.SetValue(round(self.value[3], self.wei) / self.Graduation)
                    self.J4_val.SetLabel(str(round(self.value[3], self.wei)))
                    self.joint_pos_deg[3] = self.value[3]
                    self.J5.SetValue(round(self.value[4], self.wei) / self.Graduation)
                    self.J5_val.SetLabel(str(round(self.value[4], self.wei)))
                    self.joint_pos_deg[4] = self.value[4]
                    self.J6.SetValue(round(self.value[5], self.wei) / self.Graduation)
                    self.J6_val.SetLabel(str(round(self.value[5], self.wei)))
                    self.joint_pos_deg[5] = self.value[5]
                else:
                    self.Dis_npz_state.SetLabel("             This key is not exist!")
                    self.Dis_npz_state.SetForegroundColour("red")

            else:
                self.Dis_npz_state.SetLabel("           Enter num error!")
                self.Dis_npz_state.SetForegroundColour("red")
        else:
            self.Dis_npz_state.SetLabel("           Enter num error!")
            self.Dis_npz_state.SetForegroundColour("red")

    def OnSetTgtPts(self, event):
        self.key = self.Cordition.GetValue()
        if len(self.key) == 1:
            if ord(self.key) >= 48 and ord(self.key) < 57 and len(self.key) == 1:
                self.value = self.Communication.Get_angle()
                # self.value = np.array(shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 52, 24, 0), 24)[:],
                #                      dtype=float) # joint  angle
                # self.value = np.array(shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 128, 24, 0), 24)[:],
                #                       dtype=float) # tau
                self.save_to_npz(self.npzFileName, "target" + self.key, self.value)
                LOGGER.info("Setting target of condition %s ...", self.key)
                LOGGER.info(self.value)
                self.Dis_npz_state.SetLabel(
                    "             Save Target Position!    " + self.key + " : = \n       [" + str(
                        self.value[0]) + "     " + str(
                        self.value[1]) \
                    + "\n       " + str(self.value[2]) + "       " + str(self.value[3]) + "\n       " + str(
                        self.value[4]) + \
                    "         " + str(self.value[5]) + "]")
                self.Dis_npz_state.SetForegroundColour("blue")
            else:
                self.Dis_npz_state.SetLabel("           Enter num error!")
                self.Dis_npz_state.SetForegroundColour("red")
        else:
            self.Dis_npz_state.SetLabel("           Enter num error!")
            self.Dis_npz_state.SetForegroundColour("red")

    def OnSetIntPts(self, event):
        self.key = self.Cordition.GetValue()
        if len(self.key) == 1:
            if ord(self.key) >= 48 and ord(self.key) <= 57 and len(self.key) == 1:
                self.value = self.Communication.Get_angle()
                # self.value = np.array(shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 52, 24, 0), 24)[:],
                #                      dtype=float)#joint angle
                # self.value = np.array(shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 128, 24, 0), 24)[:],
                #                       dtype=float)
                self.save_to_npz(self.npzFileName, "initial" + self.key, self.value)
                LOGGER.info("Setting initial               of condition %s ...", self.key)
                LOGGER.info(self.value)
                self.Dis_npz_state.SetLabel(
                    "             Save Initial Position!    " + self.key + " : = \n       [" + str(
                        self.value[0]) + "     " + str(
                        self.value[1]) \
                    + "\n       " + str(self.value[2]) + "       " + str(self.value[3]) + "\n       " + str(
                        self.value[4]) + \
                    "         " + str(self.value[5]) + "]")
                self.Dis_npz_state.SetForegroundColour("green")
            else:
                self.Dis_npz_state.SetLabel("           Enter num error!")
                self.Dis_npz_state.SetForegroundColour("red")
        else:
            self.Dis_npz_state.SetLabel("           Enter num error!")
            self.Dis_npz_state.SetForegroundColour("red")


    def OnX1Checkbox(self, event):
        rospy.wait_for_service('/control_X1_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X1_State', SetBool)
            resp1 = Set_Bool(self.X1.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX2Checkbox(self, event):
        rospy.wait_for_service('/control_X2_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X2_State', SetBool)
            resp1 = Set_Bool(self.X2.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX3Checkbox(self, event):
        rospy.wait_for_service('/control_X3_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X3_State', SetBool)
            resp1 = Set_Bool(self.X3.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX4Checkbox(self, event):
        rospy.wait_for_service('/control_X1_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X4_State', SetBool)
            resp1 = Set_Bool(self.X4.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX5Checkbox(self, event):
        rospy.wait_for_service('/control_X1_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X5_State', SetBool)
            resp1 = Set_Bool(self.X5.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX6Checkbox(self, event):
        rospy.wait_for_service('/control_X6_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X6_State', SetBool)
            resp1 = Set_Bool(self.X6.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX7Checkbox(self, event):
        rospy.wait_for_service('/control_X7_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X7_State', SetBool)
            resp1 = Set_Bool(self.X7.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX8Checkbox(self, event):
        rospy.wait_for_service('/control_X8_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X8_State', SetBool)
            resp1 = Set_Bool(self.X8.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX9Checkbox(self, event):
        rospy.wait_for_service('/control_X9_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X9_State', SetBool)
            resp1 = Set_Bool(self.X9.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX10Checkbox(self, event):
        rospy.wait_for_service('/control_X10_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X10_State', SetBool)
            resp1 = Set_Bool(self.X10.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX11Checkbox(self, event):
        rospy.wait_for_service('/control_X11_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X11_State', SetBool)
            resp1 = Set_Bool(self.X11.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnX12Checkbox(self, event):
        rospy.wait_for_service('/control_X6_State')
        try:
            Set_Bool = rospy.ServiceProxy('/control_X12_State', SetBool)
            resp1 = Set_Bool(self.X12.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def OnModeChoice(self, event):
        if self.mode_choice.GetSelection() == 0:
            self.Communication.Select_Mode_Go()
        else:
            self.Communication.Select_Mode_Move()

    def OnElbowChoice(self, event):
        if self.ELBOW_choice.GetSelection() == 0:
            self.Communication.Select_Elbow_Above()
        else:
            self.Communication.Select_Elbow_Below()

    def onHome(self, event):
        self.J1.SetValue(0)
        self.J2.SetValue(0)
        self.J3.SetValue(0)
        self.J4.SetValue(0)
        self.J5.SetValue(0)
        self.J6.SetValue(0)
        self.J1_val.SetLabel("0")
        self.J2_val.SetLabel("0")
        self.J3_val.SetLabel("0")
        self.J4_val.SetLabel("0")
        self.J5_val.SetLabel("0")
        self.J6_val.SetLabel("0")
        self.joint_pos_deg = np.zeros(6)
        self.Communication.Pub_Angle(self.joint_pos_deg)

    def OnRun(self, event):
        self.xyzuvw[0] = self.CoordSetValueX.get_coord_value()
        self.xyzuvw[1] = self.CoordSetValueY.get_coord_value()
        self.xyzuvw[2] = self.CoordSetValueZ.get_coord_value()
        self.xyzuvw[3] = self.CoordSetValueU.get_coord_value()
        self.xyzuvw[4] = self.CoordSetValueV.get_coord_value()
        self.xyzuvw[5] = self.CoordSetValueW.get_coord_value()
        self.Communication.Pub_Position(self.xyzuvw)

    def DataUpdate(self, event):
        self.dispy_joint_pos = self.Communication.Get_angle()
        self.dispy_coord = self.Communication.Get_position()
        self.dispy_joint_vel = self.Communication.Get_velocity()
        self.Ik_state = self.Communication.Get_IkSate()
        self.ModeState = self.Communication.Get_ModeState()
        self.ElbowState = self.Communication.Get_ElbowState()
        self.percentage = self.Communication.Get_percentage()
        self.displayX.set_dis_value(round(self.dispy_coord[0], 3))
        self.displayY.set_dis_value(round(self.dispy_coord[1], 3))
        self.displayZ.set_dis_value(round(self.dispy_coord[2], 3))
        self.displayU.set_dis_value(round(self.dispy_coord[3], 3))
        self.displayV.set_dis_value(round(self.dispy_coord[4], 3))
        self.displayW.set_dis_value(round(self.dispy_coord[5], 3))
        self.displayJ1Q.set_dis_value(round(self.dispy_joint_pos[0], 3))
        self.displayJ2Q.set_dis_value(round(self.dispy_joint_pos[1], 3))
        self.displayJ3Q.set_dis_value(round(self.dispy_joint_pos[2], 3))
        self.displayJ4Q.set_dis_value(round(self.dispy_joint_pos[3], 3))
        self.displayJ5Q.set_dis_value(round(self.dispy_joint_pos[4], 3))
        self.displayJ6Q.set_dis_value(round(self.dispy_joint_pos[5], 3))
        self.displayJ1Qdot.set_dis_value(round(self.dispy_joint_vel[0], 3))
        self.displayJ2Qdot.set_dis_value(round(self.dispy_joint_vel[1], 3))
        self.displayJ3Qdot.set_dis_value(round(self.dispy_joint_vel[2], 3))
        self.displayJ4Qdot.set_dis_value(round(self.dispy_joint_vel[3], 3))
        self.displayJ5Qdot.set_dis_value(round(self.dispy_joint_vel[4], 3))
        self.displayJ6Qdot.set_dis_value(round(self.dispy_joint_vel[5], 3))
        if self.Ik_state == 0:
            self.Dis_Ik_result.SetLabel("Fail!")
            self.Dis_Ik_result.SetForegroundColour("red")
            self.Dis_Ik_result.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, True))
        else:
            self.Dis_Ik_result.SetLabel("Go!")
            self.Dis_Ik_result.SetForegroundColour("green")
            self.Dis_Ik_result.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, True))

        if self.ElbowState == 0:
            self.Dis_ELBOW.SetLabel("ABO")
            self.Dis_ELBOW.SetForegroundColour("red")
        else:
            self.Dis_ELBOW.SetLabel("BEL")
            self.Dis_ELBOW.SetForegroundColour("red")

        if self.ModeState == 0:
            self.Dis_Mode.SetLabel("Go")
            self.Dis_Mode.SetForegroundColour("red")
        else:
            self.Dis_Mode.SetLabel("Mov")
            self.Dis_Mode.SetForegroundColour("red")
        self.Dis_speed.SetLabel(str(self.percentage))


        if self.ServerX1.Get_output():
            self.X1.SetForegroundColour("red")
        else:
            self.X1.SetForegroundColour("grey")

        if self.ServerX2.Get_output():
            self.X2.SetForegroundColour("red")
        else:
            self.X2.SetForegroundColour("grey")

        if self.ServerX3.Get_output():
            self.X3.SetForegroundColour("red")
        else:
            self.X3.SetForegroundColour("grey")

        if self.ServerX4.Get_output():
            self.X4.SetForegroundColour("red")
        else:
            self.X4.SetForegroundColour("grey")

        if self.ServerX5.Get_output():
            self.X5.SetForegroundColour("red")
        else:
            self.X5.SetForegroundColour("grey")

        if self.ServerX6.Get_output():
            self.X6.SetForegroundColour("red")
        else:
            self.X6.SetForegroundColour("grey")

        if self.ServerX7.Get_output():
            self.X7.SetForegroundColour("red")
        else:
            self.X7.SetForegroundColour("grey")

        if self.ServerX8.Get_output():
            self.X8.SetForegroundColour("red")
        else:
            self.X8.SetForegroundColour("grey")

        if self.ServerX9.Get_output():
            self.X9.SetForegroundColour("red")
        else:
            self.X9.SetForegroundColour("grey")

        if self.ServerX10.Get_output():
            self.X10.SetForegroundColour("red")
        else:
            self.X10.SetForegroundColour("grey")

        if self.ServerX11.Get_output():
            self.X11.SetForegroundColour("red")
        else:
            self.X11.SetForegroundColour("grey")

        if self.ServerX12.Get_output():
            self.X12.SetForegroundColour("red")
        else:
            self.X12.SetForegroundColour("grey")

    def OnJ1SliderScroll(self, event):
        self.joint_pos_deg[0] = self.J1.GetValue() * self.Graduation
        self.J1_val.SetLabel(str(round(self.joint_pos_deg[0], self.wei)))
        self.Communication.Pub_Angle(self.joint_pos_deg)

    def OnJ2SliderScroll(self, event):
        self.joint_pos_deg[1] = self.J2.GetValue() * self.Graduation
        self.J2_val.SetLabel(str(round(self.joint_pos_deg[1], self.wei)))
        self.Communication.Pub_Angle(self.joint_pos_deg)

    def OnJ3SliderScroll(self, event):
        self.joint_pos_deg[2] = self.J3.GetValue() * self.Graduation
        self.J3_val.SetLabel(str(round(self.joint_pos_deg[2], self.wei)))
        self.Communication.Pub_Angle(self.joint_pos_deg)

    def OnJ4SliderScroll(self, event):
        self.joint_pos_deg[3] = self.J4.GetValue() * self.Graduation
        self.J4_val.SetLabel(str(round(self.joint_pos_deg[3], self.wei)))
        self.Communication.Pub_Angle(self.joint_pos_deg)

    def OnJ5SliderScroll(self, event):
        self.joint_pos_deg[4] = self.J5.GetValue() * self.Graduation
        self.J5_val.SetLabel(str(round(self.joint_pos_deg[4], self.wei)))
        self.Communication.Pub_Angle(self.joint_pos_deg)

    def OnJ6SliderScroll(self, event):
        self.joint_pos_deg[5] = self.J6.GetValue() * self.Graduation
        self.J6_val.SetLabel(str(round(self.joint_pos_deg[5], self.wei)))
        self.Communication.Pub_Angle(self.joint_pos_deg)

    def OnperiodSliderScroll(self, event):
        self.period_val.SetLabel(str(self.period.GetValue()))
        self.Communication.Set_Send_percentage(self.period.GetValue())


class coordnate():
    def __init__(self, pnl, coord_label, value, size_x, size_y, offset_x, offset_y):
        wx.StaticText(pnl, label=coord_label, pos=(offset_x - 25, offset_y))
        self.txt = wx.TextCtrl(pnl, -1, str(value), size=(size_x, size_y), pos=(offset_x, offset_y))

    def get_coord_value(self):
        return self.txt.GetValue()


class display_pos():
    def __init__(self, pnl, dis_label, value, offset_x, offset_y):
        self.dis_label = wx.StaticText(pnl, label=dis_label, pos=(offset_x - 25, offset_y))
        self.dis_value = wx.StaticText(pnl, label=str(value), pos=(offset_x, offset_y))
        self.dis_label.SetForegroundColour("blue")

    def set_dis_value(self, dis_value):
        self.dis_value.SetLabel(str(dis_value))

class CheckBox(object):
    """docstring for CheckBox"""
    def __init__(self,Panel, input_name, pos_x, pos_y, rosserver_name):
        super(CheckBox, self).__init__()
        self.X = wx.CheckBox(Panel, -1, input_name, pos=(pos_x,pos_y), size=(50, 30))
        self.Y = wx.StaticText(Panel, label= ".", pos=(pos_x+50, pos_y-75))
        self.Y.SetFont(wx.Font(80, wx.DECORATIVE, wx.NORMAL, wx.BOLD))
        self.Y.SetForegroundColour("grey")
        self.X.Bind(wx.EVT_CHECKBOX, self.OnXCheckbox)
        self.ServiceName = rosserver_name
    def OnXCheckbox(self, event):
        rospy.wait_for_service(self.ServiceName)
        try:
            Set_Bool = rospy.ServiceProxy(self.ServiceName, SetBool)
            resp1 = Set_Bool(self.X.GetValue())
            return resp1.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def SetForegroundColour(self, color):
        self.Y.SetForegroundColour(color)

    def GetValue(self):
        return self.X.GetValue()


if __name__ == '__main__':
    try:
        ex = wx.App()
        ControlPanel(None)
        ex.MainLoop()
    except rospy.ROSInterruptException:
        pass
