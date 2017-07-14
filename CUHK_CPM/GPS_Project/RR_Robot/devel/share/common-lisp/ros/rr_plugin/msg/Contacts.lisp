; Auto-generated. Do not edit!


(cl:in-package rr_plugin-msg)


;//! \htmlinclude Contacts.msg.html

(cl:defclass <Contacts> (roslisp-msg-protocol:ros-message)
  ((Contacts
    :reader Contacts
    :initarg :Contacts
    :type (cl:vector rr_plugin-msg:Contact)
   :initform (cl:make-array 0 :element-type 'rr_plugin-msg:Contact :initial-element (cl:make-instance 'rr_plugin-msg:Contact))))
)

(cl:defclass Contacts (<Contacts>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Contacts>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Contacts)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-msg:<Contacts> is deprecated: use rr_plugin-msg:Contacts instead.")))

(cl:ensure-generic-function 'Contacts-val :lambda-list '(m))
(cl:defmethod Contacts-val ((m <Contacts>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-msg:Contacts-val is deprecated.  Use rr_plugin-msg:Contacts instead.")
  (Contacts m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Contacts>) ostream)
  "Serializes a message object of type '<Contacts>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'Contacts))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'Contacts))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Contacts>) istream)
  "Deserializes a message object of type '<Contacts>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'Contacts) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'Contacts)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'rr_plugin-msg:Contact))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Contacts>)))
  "Returns string type for a message object of type '<Contacts>"
  "rr_plugin/Contacts")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Contacts)))
  "Returns string type for a message object of type 'Contacts"
  "rr_plugin/Contacts")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Contacts>)))
  "Returns md5sum for a message object of type '<Contacts>"
  "d7aa4d50bde603c6f4ee8de785620b01")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Contacts)))
  "Returns md5sum for a message object of type 'Contacts"
  "d7aa4d50bde603c6f4ee8de785620b01")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Contacts>)))
  "Returns full string definition for message of type '<Contacts>"
  (cl:format cl:nil "rr_plugin/Contact[] Contacts~%~%================================================================================~%MSG: rr_plugin/Contact~%string collision1~%string collision2~%geometry_msgs/Vector3[] position~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Contacts)))
  "Returns full string definition for message of type 'Contacts"
  (cl:format cl:nil "rr_plugin/Contact[] Contacts~%~%================================================================================~%MSG: rr_plugin/Contact~%string collision1~%string collision2~%geometry_msgs/Vector3[] position~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Contacts>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'Contacts) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Contacts>))
  "Converts a ROS message object to a list"
  (cl:list 'Contacts
    (cl:cons ':Contacts (Contacts msg))
))
