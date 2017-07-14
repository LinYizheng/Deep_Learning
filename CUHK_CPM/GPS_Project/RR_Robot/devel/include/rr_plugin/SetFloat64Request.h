// Generated by gencpp from file rr_plugin/SetFloat64Request.msg
// DO NOT EDIT!


#ifndef RR_PLUGIN_MESSAGE_SETFLOAT64REQUEST_H
#define RR_PLUGIN_MESSAGE_SETFLOAT64REQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace rr_plugin
{
template <class ContainerAllocator>
struct SetFloat64Request_
{
  typedef SetFloat64Request_<ContainerAllocator> Type;

  SetFloat64Request_()
    : data(0.0)  {
    }
  SetFloat64Request_(const ContainerAllocator& _alloc)
    : data(0.0)  {
  (void)_alloc;
    }



   typedef double _data_type;
  _data_type data;




  typedef boost::shared_ptr< ::rr_plugin::SetFloat64Request_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rr_plugin::SetFloat64Request_<ContainerAllocator> const> ConstPtr;

}; // struct SetFloat64Request_

typedef ::rr_plugin::SetFloat64Request_<std::allocator<void> > SetFloat64Request;

typedef boost::shared_ptr< ::rr_plugin::SetFloat64Request > SetFloat64RequestPtr;
typedef boost::shared_ptr< ::rr_plugin::SetFloat64Request const> SetFloat64RequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rr_plugin::SetFloat64Request_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace rr_plugin

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'rr_plugin': ['/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg'], 'geometry_msgs': ['/opt/ros/indigo/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rr_plugin::SetFloat64Request_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rr_plugin::SetFloat64Request_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rr_plugin::SetFloat64Request_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
{
  static const char* value()
  {
    return "fdb28210bfa9d7c91146260178d9a584";
  }

  static const char* value(const ::rr_plugin::SetFloat64Request_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xfdb28210bfa9d7c9ULL;
  static const uint64_t static_value2 = 0x1146260178d9a584ULL;
};

template<class ContainerAllocator>
struct DataType< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rr_plugin/SetFloat64Request";
  }

  static const char* value(const ::rr_plugin::SetFloat64Request_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float64 data\n\
\n\
";
  }

  static const char* value(const ::rr_plugin::SetFloat64Request_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.data);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SetFloat64Request_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rr_plugin::SetFloat64Request_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rr_plugin::SetFloat64Request_<ContainerAllocator>& v)
  {
    s << indent << "data: ";
    Printer<double>::stream(s, indent + "  ", v.data);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RR_PLUGIN_MESSAGE_SETFLOAT64REQUEST_H
