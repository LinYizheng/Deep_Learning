; Auto-generated. Do not edit!


(cl:in-package rr_plugin-srv)


;//! \htmlinclude GetGripperData-request.msg.html

(cl:defclass <GetGripperData-request> (roslisp-msg-protocol:ros-message)
  ((isDegree
    :reader isDegree
    :initarg :isDegree
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass GetGripperData-request (<GetGripperData-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetGripperData-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetGripperData-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<GetGripperData-request> is deprecated: use rr_plugin-srv:GetGripperData-request instead.")))

(cl:ensure-generic-function 'isDegree-val :lambda-list '(m))
(cl:defmethod isDegree-val ((m <GetGripperData-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:isDegree-val is deprecated.  Use rr_plugin-srv:isDegree instead.")
  (isDegree m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetGripperData-request>) ostream)
  "Serializes a message object of type '<GetGripperData-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'isDegree) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetGripperData-request>) istream)
  "Deserializes a message object of type '<GetGripperData-request>"
    (cl:setf (cl:slot-value msg 'isDegree) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetGripperData-request>)))
  "Returns string type for a service object of type '<GetGripperData-request>"
  "rr_plugin/GetGripperDataRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetGripperData-request)))
  "Returns string type for a service object of type 'GetGripperData-request"
  "rr_plugin/GetGripperDataRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetGripperData-request>)))
  "Returns md5sum for a message object of type '<GetGripperData-request>"
  "41a15d5585d2a91cea5a35ba0f2c1927")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetGripperData-request)))
  "Returns md5sum for a message object of type 'GetGripperData-request"
  "41a15d5585d2a91cea5a35ba0f2c1927")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetGripperData-request>)))
  "Returns full string definition for message of type '<GetGripperData-request>"
  (cl:format cl:nil "bool isDegree~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetGripperData-request)))
  "Returns full string definition for message of type 'GetGripperData-request"
  (cl:format cl:nil "bool isDegree~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetGripperData-request>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetGripperData-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetGripperData-request
    (cl:cons ':isDegree (isDegree msg))
))
;//! \htmlinclude GetGripperData-response.msg.html

(cl:defclass <GetGripperData-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (leftfingerpos
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

(cl:defclass GetGripperData-response (<GetGripperData-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetGripperData-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetGripperData-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rr_plugin-srv:<GetGripperData-response> is deprecated: use rr_plugin-srv:GetGripperData-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <GetGripperData-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:success-val is deprecated.  Use rr_plugin-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'leftfingerpos-val :lambda-list '(m))
(cl:defmethod leftfingerpos-val ((m <GetGripperData-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:leftfingerpos-val is deprecated.  Use rr_plugin-srv:leftfingerpos instead.")
  (leftfingerpos m))

(cl:ensure-generic-function 'rightfingerpos-val :lambda-list '(m))
(cl:defmethod rightfingerpos-val ((m <GetGripperData-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rr_plugin-srv:rightfingerpos-val is deprecated.  Use rr_plugin-srv:rightfingerpos instead.")
  (rightfingerpos m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetGripperData-response>) ostream)
  "Serializes a message object of type '<GetGripperData-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetGripperData-response>) istream)
  "Deserializes a message object of type '<GetGripperData-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetGripperData-response>)))
  "Returns string type for a service object of type '<GetGripperData-response>"
  "rr_plugin/GetGripperDataResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetGripperData-response)))
  "Returns string type for a service object of type 'GetGripperData-response"
  "rr_plugin/GetGripperDataResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetGripperData-response>)))
  "Returns md5sum for a message object of type '<GetGripperData-response>"
  "41a15d5585d2a91cea5a35ba0f2c1927")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetGripperData-response)))
  "Returns md5sum for a message object of type 'GetGripperData-response"
  "41a15d5585d2a91cea5a35ba0f2c1927")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetGripperData-response>)))
  "Returns full string definition for message of type '<GetGripperData-response>"
  (cl:format cl:nil "bool success~%float64 leftfingerpos~%float64 rightfingerpos~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetGripperData-response)))
  "Returns full string definition for message of type 'GetGripperData-response"
  (cl:format cl:nil "bool success~%float64 leftfingerpos~%float64 rightfingerpos~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetGripperData-response>))
  (cl:+ 0
     1
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetGripperData-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetGripperData-response
    (cl:cons ':success (success msg))
    (cl:cons ':leftfingerpos (leftfingerpos msg))
    (cl:cons ':rightfingerpos (rightfingerpos msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetGripperData)))
  'GetGripperData-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetGripperData)))
  'GetGripperData-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetGripperData)))
  "Returns string type for a service object of type '<GetGripperData>"
  "rr_plugin/GetGripperData")