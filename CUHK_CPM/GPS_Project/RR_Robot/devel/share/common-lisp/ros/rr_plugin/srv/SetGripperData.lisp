; Auto-generated. Do not edit!


(cl:in-package rr_plugin-srv)


;//! \htmlinclude SetGripperData-request.msg.html

(cl:defclass <SetGripperData-request> (roslisp-msg-protocol:ros-message)
  ((leftfingerpos
    :reader leftfingerpos
    :initarg :leftfingerpos
    :type cl:float
    :initform 0.0)
   (rightfingerpos
    :reader rightfingerpos
    :initarg :rightfingerpos
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetGripperData-request (<SetGripperData-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetGripperData-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetGripperData-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<SetGripperData-request> is deprecated: use rr_plugin-srv:SetGripperData-request instead.")))

(cl:ensure-generic-function 'leftfingerpos-val :lambda-list '(m))
(cl:defmethod leftfingerpos-val ((m <SetGripperData-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:leftfingerpos-val is deprecated.  Use rr_plugin-srv:leftfingerpos instead.")
  (leftfingerpos m))

(cl:ensure-generic-function 'rightfingerpos-val :lambda-list '(m))
(cl:defmethod rightfingerpos-val ((m <SetGripperData-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:rightfingerpos-val is deprecated.  Use rr_plugin-srv:rightfingerpos instead.")
  (rightfingerpos m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetGripperData-request>) ostream)
  "Serializes a message object of type '<SetGripperData-request>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'leftfingerpos))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'rightfingerpos))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetGripperData-request>) istream)
  "Deserializes a message object of type '<SetGripperData-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'leftfingerpos) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rightfingerpos) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetGripperData-request>)))
  "Returns string type for a service object of type '<SetGripperData-request>"
  "rr_plugin/SetGripperDataRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetGripperData-request)))
  "Returns string type for a service object of type 'SetGripperData-request"
  "rr_plugin/SetGripperDataRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetGripperData-request>)))
  "Returns md5sum for a message object of type '<SetGripperData-request>"
  "ac17c576c6826b5f5e8daea06d6ed830")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetGripperData-request)))
  "Returns md5sum for a message object of type 'SetGripperData-request"
  "ac17c576c6826b5f5e8daea06d6ed830")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetGripperData-request>)))
  "Returns full string definition for message of type '<SetGripperData-request>"
  (cl:format cl:nil "float64 leftfingerpos~%float64 rightfingerpos~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetGripperData-request)))
  "Returns full string definition for message of type 'SetGripperData-request"
  (cl:format cl:nil "float64 leftfingerpos~%float64 rightfingerpos~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetGripperData-request>))
  (cl:+ 0
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetGripperData-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetGripperData-request
    (cl:cons ':leftfingerpos (leftfingerpos msg))
    (cl:cons ':rightfingerpos (rightfingerpos msg))
))
;//! \htmlinclude SetGripperData-response.msg.html

(cl:defclass <SetGripperData-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetGripperData-response (<SetGripperData-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetGripperData-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetGripperData-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<SetGripperData-response> is deprecated: use rr_plugin-srv:SetGripperData-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetGripperData-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:success-val is deprecated.  Use rr_plugin-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetGripperData-response>) ostream)
  "Serializes a message object of type '<SetGripperData-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetGripperData-response>) istream)
  "Deserializes a message object of type '<SetGripperData-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetGripperData-response>)))
  "Returns string type for a service object of type '<SetGripperData-response>"
  "rr_plugin/SetGripperDataResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetGripperData-response)))
  "Returns string type for a service object of type 'SetGripperData-response"
  "rr_plugin/SetGripperDataResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetGripperData-response>)))
  "Returns md5sum for a message object of type '<SetGripperData-response>"
  "ac17c576c6826b5f5e8daea06d6ed830")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetGripperData-response)))
  "Returns md5sum for a message object of type 'SetGripperData-response"
  "ac17c576c6826b5f5e8daea06d6ed830")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetGripperData-response>)))
  "Returns full string definition for message of type '<SetGripperData-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetGripperData-response)))
  "Returns full string definition for message of type 'SetGripperData-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetGripperData-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetGripperData-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetGripperData-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetGripperData)))
  'SetGripperData-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetGripperData)))
  'SetGripperData-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetGripperData)))
  "Returns string type for a service object of type '<SetGripperData>"
  "rr_plugin/SetGripperData")