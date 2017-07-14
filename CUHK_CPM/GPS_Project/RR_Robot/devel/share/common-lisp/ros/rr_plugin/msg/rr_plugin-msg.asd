
(cl:in-package :asdf)

(defsystem "rr_plugin-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "Contacts" :depends-on ("_package_Contacts"))
    (:file "_package_Contacts" :depends-on ("_package"))
    (:file "Contact" :depends-on ("_package_Contact"))
    (:file "_package_Contact" :depends-on ("_package"))
  ))