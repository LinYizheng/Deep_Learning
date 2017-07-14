// Generated by gencpp from file rr_plugin/SetFloat64.msg
// DO NOT EDIT!


#ifndef RR_PLUGIN_MESSAGE_SETFLOAT64_H
#define RR_PLUGIN_MESSAGE_SETFLOAT64_H

#include <ros/service_traits.h>


#include <rr_plugin/SetFloat64Request.h>
#include <rr_plugin/SetFloat64Response.h>


namespace rr_plugin
{

struct SetFloat64
{

typedef SetFloat64Request Request;
typedef SetFloat64Response Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct SetFloat64
} // namespace rr_plugin


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::rr_plugin::SetFloat64 > {
  static const char* value()
  {
    return "6dffcb6acc6bec80315e1c470ea1bca9";
  }

  static const char* value(const ::rr_plugin::SetFloat64&) { return value(); }
};

template<>
struct DataType< ::rr_plugin::SetFloat64 > {
  static const char* value()
  {
    return "rr_plugin/SetFloat64";
  }

  static const char* value(const ::rr_plugin::SetFloat64&) { return value(); }
};


// service_traits::MD5Sum< ::rr_plugin::SetFloat64Request> should match 
// service_traits::MD5Sum< ::rr_plugin::SetFloat64 > 
template<>
struct MD5Sum< ::rr_plugin::SetFloat64Request>
{
  static const char* value()
  {
    return MD5Sum< ::rr_plugin::SetFloat64 >::value();
  }
  static const char* value(const ::rr_plugin::SetFloat64Request&)
  {
    return value();
  }
};

// service_traits::DataType< ::rr_plugin::SetFloat64Request> should match 
// service_traits::DataType< ::rr_plugin::SetFloat64 > 
template<>
struct DataType< ::rr_plugin::SetFloat64Request>
{
  static const char* value()
  {
    return DataType< ::rr_plugin::SetFloat64 >::value();
  }
  static const char* value(const ::rr_plugin::SetFloat64Request&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::rr_plugin::SetFloat64Response> should match 
// service_traits::MD5Sum< ::rr_plugin::SetFloat64 > 
template<>
struct MD5Sum< ::rr_plugin::SetFloat64Response>
{
  static const char* value()
  {
    return MD5Sum< ::rr_plugin::SetFloat64 >::value();
  }
  static const char* value(const ::rr_plugin::SetFloat64Response&)
  {
    return value();
  }
};

// service_traits::DataType< ::rr_plugin::SetFloat64Response> should match 
// service_traits::DataType< ::rr_plugin::SetFloat64 > 
template<>
struct DataType< ::rr_plugin::SetFloat64Response>
{
  static const char* value()
  {
    return DataType< ::rr_plugin::SetFloat64 >::value();
  }
  static const char* value(const ::rr_plugin::SetFloat64Response&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // RR_PLUGIN_MESSAGE_SETFLOAT64_H
