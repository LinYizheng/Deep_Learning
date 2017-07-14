# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "rr_plugin: 2 messages, 7 services")

set(MSG_I_FLAGS "-Irr_plugin:/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg;-Igeometry_msgs:/opt/ros/indigo/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/indigo/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(genlisp REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(rr_plugin_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv" ""
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg" "geometry_msgs/Vector3:rr_plugin/Contact"
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv" "geometry_msgs/Point"
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv" ""
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv" ""
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg" "geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv" ""
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv" ""
)

get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv" NAME_WE)
add_custom_target(_rr_plugin_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rr_plugin" "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv" ""
)

#
#  langs = gencpp;genlisp;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)
_generate_msg_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Vector3.msg;/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)

### Generating Services
_generate_srv_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)
_generate_srv_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)
_generate_srv_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)
_generate_srv_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)
_generate_srv_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)
_generate_srv_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)
_generate_srv_cpp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
)

### Generating Module File
_generate_module_cpp(rr_plugin
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(rr_plugin_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(rr_plugin_generate_messages rr_plugin_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_cpp _rr_plugin_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rr_plugin_gencpp)
add_dependencies(rr_plugin_gencpp rr_plugin_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rr_plugin_generate_messages_cpp)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)
_generate_msg_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Vector3.msg;/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)

### Generating Services
_generate_srv_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)
_generate_srv_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)
_generate_srv_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)
_generate_srv_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)
_generate_srv_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)
_generate_srv_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)
_generate_srv_lisp(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
)

### Generating Module File
_generate_module_lisp(rr_plugin
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(rr_plugin_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(rr_plugin_generate_messages rr_plugin_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_lisp _rr_plugin_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rr_plugin_genlisp)
add_dependencies(rr_plugin_genlisp rr_plugin_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rr_plugin_generate_messages_lisp)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)
_generate_msg_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Vector3.msg;/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)

### Generating Services
_generate_srv_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)
_generate_srv_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/indigo/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)
_generate_srv_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)
_generate_srv_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)
_generate_srv_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)
_generate_srv_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)
_generate_srv_py(rr_plugin
  "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
)

### Generating Module File
_generate_module_py(rr_plugin
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(rr_plugin_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(rr_plugin_generate_messages rr_plugin_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/DelGroup.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contacts.msg" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetPoint.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetGripperData.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/AddGroup.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg/Contact.msg" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/GetGripperData.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/srv/SetFloat64MultiArray.srv" NAME_WE)
add_dependencies(rr_plugin_generate_messages_py _rr_plugin_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rr_plugin_genpy)
add_dependencies(rr_plugin_genpy rr_plugin_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rr_plugin_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rr_plugin
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(rr_plugin_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(rr_plugin_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rr_plugin
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(rr_plugin_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(rr_plugin_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rr_plugin
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(rr_plugin_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(rr_plugin_generate_messages_py std_msgs_generate_messages_py)
endif()
