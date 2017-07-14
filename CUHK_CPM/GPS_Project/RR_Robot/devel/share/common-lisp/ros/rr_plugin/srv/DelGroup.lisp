; Auto-generated. Do not edit!


(cl:in-package rr_plugin-srv)


;//! \htmlinclude DelGroup-request.msg.html

(cl:defclass <DelGroup-request> (roslisp-msg-protocol:ros-message)
  ((type
    :reader type
    :initarg :type
    :type cl:string
    :initform "")
   (index
    :reader index
    :initarg :index
    :type cl:fixnum
    :initform 0))
)

(cl:defclass DelGroup-request (<DelGroup-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DelGroup-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DelGroup-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<DelGroup-request> is deprecated: use rr_plugin-srv:DelGroup-request instead.")))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <DelGroup-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:type-val is deprecated.  Use rr_plugin-srv:type instead.")
  (type m))

(cl:ensure-generic-function 'index-val :lambda-list '(m))
(cl:defmethod index-val ((m <DelGroup-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:index-val is deprecated.  Use rr_plugin-srv:index instead.")
  (index m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DelGroup-request>) ostream)
  "Serializes a message object of type '<DelGroup-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'type))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'type))
  (cl:let* ((signed (cl:slot-value msg 'index)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DelGroup-request>) istream)
  "Deserializes a message object of type '<DelGroup-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'type) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'type) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'index) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DelGroup-request>)))
  "Returns string type for a service object of type '<DelGroup-request>"
  "rr_plugin/DelGroupRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DelGroup-request)))
  "Returns string type for a service object of type 'DelGroup-request"
  "rr_plugin/DelGroupRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DelGroup-request>)))
  "Returns md5sum for a message object of type '<DelGroup-request>"
  "7d90912713b548daf0792e10bc03054b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DelGroup-request)))
  "Returns md5sum for a message object of type 'DelGroup-request"
  "7d90912713b548daf0792e10bc03054b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DelGroup-request>)))
  "Returns full string definition for message of type '<DelGroup-request>"
  (cl:format cl:nil "string type~%int8 index~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DelGroup-request)))
  "Returns full string definition for message of type 'DelGroup-request"
  (cl:format cl:nil "string type~%int8 index~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DelGroup-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'type))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DelGroup-request>))
  "Converts a ROS message object to a list"
  (cl:list 'DelGroup-request
    (cl:cons ':type (type msg))
    (cl:cons ':index (index msg))
))
;//! \htmlinclude DelGroup-response.msg.html

(cl:defclass <DelGroup-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass DelGroup-response (<DelGroup-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DelGroup-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DelGroup-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<DelGroup-response> is deprecated: use rr_plugin-srv:DelGroup-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <DelGroup-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:success-val is deprecated.  Use rr_plugin-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DelGroup-response>) ostream)
  "Serializes a message object of type '<DelGroup-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DelGroup-response>) istream)
  "Deserializes a message object of type '<DelGroup-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DelGroup-response>)))
  "Returns string type for a service object of type '<DelGroup-response>"
  "rr_plugin/DelGroupResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DelGroup-response)))
  "Returns string type for a service object of type 'DelGroup-response"
  "rr_plugin/DelGroupResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DelGroup-response>)))
  "Returns md5sum for a message object of type '<DelGroup-response>"
  "7d90912713b548daf0792e10bc03054b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DelGroup-response)))
  "Returns md5sum for a message object of type 'DelGroup-response"
  "7d90912713b548daf0792e10bc03054b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DelGroup-response>)))
  "Returns full string definition for message of type '<DelGroup-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DelGroup-response)))
  "Returns full string definition for message of type 'DelGroup-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DelGroup-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DelGroup-response>))
  "Converts a ROS message object to a list"
  (cl:list 'DelGroup-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'DelGroup)))
  'DelGroup-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'DelGroup)))
  'DelGroup-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DelGroup)))
  "Returns string type for a service object of type '<DelGroup>"
  "rr_plugin/DelGroup")