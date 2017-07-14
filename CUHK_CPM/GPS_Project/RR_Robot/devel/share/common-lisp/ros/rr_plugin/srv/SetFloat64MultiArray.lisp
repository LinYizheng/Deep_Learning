; Auto-generated. Do not edit!


(cl:in-package rr_plugin-srv)


;//! \htmlinclude SetFloat64MultiArray-request.msg.html

(cl:defclass <SetFloat64MultiArray-request> (roslisp-msg-protocol:ros-message)
  ((data
    :reader data
    :initarg :data
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass SetFloat64MultiArray-request (<SetFloat64MultiArray-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetFloat64MultiArray-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetFloat64MultiArray-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<SetFloat64MultiArray-request> is deprecated: use rr_plugin-srv:SetFloat64MultiArray-request instead.")))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <SetFloat64MultiArray-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:data-val is deprecated.  Use rr_plugin-srv:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetFloat64MultiArray-request>) ostream)
  "Serializes a message object of type '<SetFloat64MultiArray-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetFloat64MultiArray-request>) istream)
  "Deserializes a message object of type '<SetFloat64MultiArray-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetFloat64MultiArray-request>)))
  "Returns string type for a service object of type '<SetFloat64MultiArray-request>"
  "rr_plugin/SetFloat64MultiArrayRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetFloat64MultiArray-request)))
  "Returns string type for a service object of type 'SetFloat64MultiArray-request"
  "rr_plugin/SetFloat64MultiArrayRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetFloat64MultiArray-request>)))
  "Returns md5sum for a message object of type '<SetFloat64MultiArray-request>"
  "c63bf8592d64621eba801bc64f49f640")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetFloat64MultiArray-request)))
  "Returns md5sum for a message object of type 'SetFloat64MultiArray-request"
  "c63bf8592d64621eba801bc64f49f640")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetFloat64MultiArray-request>)))
  "Returns full string definition for message of type '<SetFloat64MultiArray-request>"
  (cl:format cl:nil "float64[]         data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetFloat64MultiArray-request)))
  "Returns full string definition for message of type 'SetFloat64MultiArray-request"
  (cl:format cl:nil "float64[]         data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetFloat64MultiArray-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetFloat64MultiArray-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetFloat64MultiArray-request
    (cl:cons ':data (data msg))
))
;//! \htmlinclude SetFloat64MultiArray-response.msg.html

(cl:defclass <SetFloat64MultiArray-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (result
    :reader result
    :initarg :result
    :type cl:integer
    :initform 0))
)

(cl:defclass SetFloat64MultiArray-response (<SetFloat64MultiArray-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetFloat64MultiArray-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetFloat64MultiArray-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<SetFloat64MultiArray-response> is deprecated: use rr_plugin-srv:SetFloat64MultiArray-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetFloat64MultiArray-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:success-val is deprecated.  Use rr_plugin-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <SetFloat64MultiArray-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:result-val is deprecated.  Use rr_plugin-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetFloat64MultiArray-response>) ostream)
  "Serializes a message object of type '<SetFloat64MultiArray-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'result)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetFloat64MultiArray-response>) istream)
  "Deserializes a message object of type '<SetFloat64MultiArray-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'result) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetFloat64MultiArray-response>)))
  "Returns string type for a service object of type '<SetFloat64MultiArray-response>"
  "rr_plugin/SetFloat64MultiArrayResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetFloat64MultiArray-response)))
  "Returns string type for a service object of type 'SetFloat64MultiArray-response"
  "rr_plugin/SetFloat64MultiArrayResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetFloat64MultiArray-response>)))
  "Returns md5sum for a message object of type '<SetFloat64MultiArray-response>"
  "c63bf8592d64621eba801bc64f49f640")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetFloat64MultiArray-response)))
  "Returns md5sum for a message object of type 'SetFloat64MultiArray-response"
  "c63bf8592d64621eba801bc64f49f640")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetFloat64MultiArray-response>)))
  "Returns full string definition for message of type '<SetFloat64MultiArray-response>"
  (cl:format cl:nil "bool success~%int32 result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetFloat64MultiArray-response)))
  "Returns full string definition for message of type 'SetFloat64MultiArray-response"
  (cl:format cl:nil "bool success~%int32 result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetFloat64MultiArray-response>))
  (cl:+ 0
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetFloat64MultiArray-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetFloat64MultiArray-response
    (cl:cons ':success (success msg))
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetFloat64MultiArray)))
  'SetFloat64MultiArray-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetFloat64MultiArray)))
  'SetFloat64MultiArray-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetFloat64MultiArray)))
  "Returns string type for a service object of type '<SetFloat64MultiArray>"
  "rr_plugin/SetFloat64MultiArray")