
(cl:in-package :asdf)

(defsystem "rr_plugin-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "AddGroup" :depends-on ("_package_AddGroup"))
    (:file "_package_AddGroup" :depends-on ("_package"))
    (:file "SetFloat64" :depends-on ("_package_SetFloat64"))
    (:file "_package_SetFloat64" :depends-on ("_package"))
    (:file "SetPoint" :depends-on ("_package_SetPoint"))
    (:file "_package_SetPoint" :depends-on ("_package"))
    (:file "GetGripperData" :depends-on ("_package_GetGripperData"))
    (:file "_package_GetGripperData" :depends-on ("_package"))
    (:file "DelGroup" :depends-on ("_package_DelGroup"))
    (:file "_package_DelGroup" :depends-on ("_package"))
    (:file "SetGripperData" :depends-on ("_package_SetGripperData"))
    (:file "_package_SetGripperData" :depends-on ("_package"))
    (:file "SetFloat64MultiArray" :depends-on ("_package_SetFloat64MultiArray"))
    (:file "_package_SetFloat64MultiArray" :depends-on ("_package"))
  ))