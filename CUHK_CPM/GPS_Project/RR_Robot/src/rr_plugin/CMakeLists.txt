cmake_minimum_required(VERSION 2.8.3)
project(rr_plugin)
if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
endif()
find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  gazebo_plugins
  roscpp
  std_msgs
  geometry_msgs
  message_generation
)

find_package(gazebo REQUIRED)

########################
## Message generation ##
########################

add_message_files(
   FILES
   Contact.msg
   Contacts.msg	
 )

add_service_files(
  FILES
  AddGroup.srv
  DelGroup.srv
  SetPoint.srv
  GetGripperData.srv
  SetGripperData.srv
  SetFloat64.srv
  SetFloat64MultiArray.srv
  
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  message_generation
)

include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
