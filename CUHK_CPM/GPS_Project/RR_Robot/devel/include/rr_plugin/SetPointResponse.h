// Generated by gencpp from file rr_plugin/SetPointResponse.msg
// DO NOT EDIT!


#ifndef RR_PLUGIN_MESSAGE_SETPOINTRESPONSE_H
#define RR_PLUGIN_MESSAGE_SETPOINTRESPONSE_H


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
struct SetPointResponse_
{
  typedef SetPointResponse_<ContainerAllocator> Type;

  SetPointResponse_()
    : success(false)  {
    }
  SetPointResponse_(const ContainerAllocator& _alloc)
    : success(false)  {
  (void)_alloc;
    }



   typedef uint8_t _success_type;
  _success_type success;




  typedef boost::shared_ptr< ::rr_plugin::SetPointResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rr_plugin::SetPointResponse_<ContainerAllocator> const> ConstPtr;

}; // struct SetPointResponse_

typedef ::rr_plugin::SetPointResponse_<std::allocator<void> > SetPointResponse;

typedef boost::shared_ptr< ::rr_plugin::SetPointResponse > SetPointResponsePtr;
typedef boost::shared_ptr< ::rr_plugin::SetPointResponse const> SetPointResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rr_plugin::SetPointResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rr_plugin::SetPointResponse_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rr_plugin::SetPointResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rr_plugin::SetPointResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rr_plugin::SetPointResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "358e233cde0c8a8bcfea4ce193f8fc15";
  }

  static const char* value(const ::rr_plugin::SetPointResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x358e233cde0c8a8bULL;
  static const uint64_t static_value2 = 0xcfea4ce193f8fc15ULL;
};

template<class ContainerAllocator>
struct DataType< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rr_plugin/SetPointResponse";
  }

  static const char* value(const ::rr_plugin::SetPointResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "bool 	success\n\
\n\
";
  }

  static const char* value(const ::rr_plugin::SetPointResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.success);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SetPointResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rr_plugin::SetPointResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rr_plugin::SetPointResponse_<ContainerAllocator>& v)
  {
    s << indent << "success: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.success);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RR_PLUGIN_MESSAGE_SETPOINTRESPONSE_H
