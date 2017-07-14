; Auto-generated. Do not edit!


(cl:in-package rr_plugin-srv)


;//! \htmlinclude SetPoint-request.msg.html

(cl:defclass <SetPoint-request> (roslisp-msg-protocol:ros-message)
  ((type
    :reader type
    :initarg :type
    :type cl:string
    :initform "")
   (index
    :reader index
    :initarg :index
    :type cl:fixnum
    :initform 0)
   (point
    :reader point
    :initarg :point
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point)))
)

(cl:defclass SetPoint-request (<SetPoint-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetPoint-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetPoint-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<SetPoint-request> is deprecated: use rr_plugin-srv:SetPoint-request instead.")))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <SetPoint-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:type-val is deprecated.  Use rr_plugin-srv:type instead.")
  (type m))

(cl:ensure-generic-function 'index-val :lambda-list '(m))
(cl:defmethod index-val ((m <SetPoint-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:index-val is deprecated.  Use rr_plugin-srv:index instead.")
  (index m))

(cl:ensure-generic-function 'point-val :lambda-list '(m))
(cl:defmethod point-val ((m <SetPoint-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:point-val is deprecated.  Use rr_plugin-srv:point instead.")
  (point m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetPoint-request>) ostream)
  "Serializes a message object of type '<SetPoint-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'type))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'type))
  (cl:let* ((signed (cl:slot-value msg 'index)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'point) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetPoint-request>) istream)
  "Deserializes a message object of type '<SetPoint-request>"
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
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'point) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetPoint-request>)))
  "Returns string type for a service object of type '<SetPoint-request>"
  "rr_plugin/SetPointRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetPoint-request)))
  "Returns string type for a service object of type 'SetPoint-request"
  "rr_plugin/SetPointRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetPoint-request>)))
  "Returns md5sum for a message object of type '<SetPoint-request>"
  "8b2f9f6b13ea2dce923aca7be2a42f79")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetPoint-request)))
  "Returns md5sum for a message object of type 'SetPoint-request"
  "8b2f9f6b13ea2dce923aca7be2a42f79")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetPoint-request>)))
  "Returns full string definition for message of type '<SetPoint-request>"
  (cl:format cl:nil "string type~%int8 index~%geometry_msgs/Point point~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetPoint-request)))
  "Returns full string definition for message of type 'SetPoint-request"
  (cl:format cl:nil "string type~%int8 index~%geometry_msgs/Point point~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetPoint-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'type))
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'point))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetPoint-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetPoint-request
    (cl:cons ':type (type msg))
    (cl:cons ':index (index msg))
    (cl:cons ':point (point msg))
))
;//! \htmlinclude SetPoint-response.msg.html

(cl:defclass <SetPoint-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetPoint-response (<SetPoint-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetPoint-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetPoint-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<SetPoint-response> is deprecated: use rr_plugin-srv:SetPoint-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetPoint-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:success-val is deprecated.  Use rr_plugin-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetPoint-response>) ostream)
  "Serializes a message object of type '<SetPoint-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetPoint-response>) istream)
  "Deserializes a message object of type '<SetPoint-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetPoint-response>)))
  "Returns string type for a service object of type '<SetPoint-response>"
  "rr_plugin/SetPointResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetPoint-response)))
  "Returns string type for a service object of type 'SetPoint-response"
  "rr_plugin/SetPointResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetPoint-response>)))
  "Returns md5sum for a message object of type '<SetPoint-response>"
  "8b2f9f6b13ea2dce923aca7be2a42f79")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetPoint-response)))
  "Returns md5sum for a message object of type 'SetPoint-response"
  "8b2f9f6b13ea2dce923aca7be2a42f79")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetPoint-response>)))
  "Returns full string definition for message of type '<SetPoint-response>"
  (cl:format cl:nil "bool 	success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetPoint-response)))
  "Returns full string definition for message of type 'SetPoint-response"
  (cl:format cl:nil "bool 	success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetPoint-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetPoint-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetPoint-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetPoint)))
  'SetPoint-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetPoint)))
  'SetPoint-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetPoint)))
  "Returns string type for a service object of type '<SetPoint>"
  "rr_plugin/SetPoint")