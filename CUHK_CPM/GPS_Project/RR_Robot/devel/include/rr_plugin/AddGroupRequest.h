// Generated by gencpp from file rr_plugin/AddGroupRequest.msg
// DO NOT EDIT!


#ifndef RR_PLUGIN_MESSAGE_ADDGROUPREQUEST_H
#define RR_PLUGIN_MESSAGE_ADDGROUPREQUEST_H


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
struct AddGroupRequest_
{
  typedef AddGroupRequest_<ContainerAllocator> Type;

  AddGroupRequest_()
    : type()
    , color()  {
    }
  AddGroupRequest_(const ContainerAllocator& _alloc)
    : type(_alloc)
    , color(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _type_type;
  _type_type type;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _color_type;
  _color_type color;




  typedef boost::shared_ptr< ::rr_plugin::AddGroupRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rr_plugin::AddGroupRequest_<ContainerAllocator> const> ConstPtr;

}; // struct AddGroupRequest_

typedef ::rr_plugin::AddGroupRequest_<std::allocator<void> > AddGroupRequest;

typedef boost::shared_ptr< ::rr_plugin::AddGroupRequest > AddGroupRequestPtr;
typedef boost::shared_ptr< ::rr_plugin::AddGroupRequest const> AddGroupRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rr_plugin::AddGroupRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace rr_plugin

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'rr_plugin': ['/home/turinglife/GPS_Project/RR_Robot/src/rr_plugin/msg'], 'geometry_msgs': ['/opt/ros/indigo/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rr_plugin::AddGroupRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rr_plugin::AddGroupRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rr_plugin::AddGroupRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "e33fd06afe7da7ba7e11dfc0bf097031";
  }

  static const char* value(const ::rr_plugin::AddGroupRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xe33fd06afe7da7baULL;
  static const uint64_t static_value2 = 0x7e11dfc0bf097031ULL;
};

template<class ContainerAllocator>
struct DataType< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rr_plugin/AddGroupRequest";
  }

  static const char* value(const ::rr_plugin::AddGroupRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string type\n\
string color\n\
";
  }

  static const char* value(const ::rr_plugin::AddGroupRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.type);
      stream.next(m.color);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct AddGroupRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rr_plugin::AddGroupRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rr_plugin::AddGroupRequest_<ContainerAllocator>& v)
  {
    s << indent << "type: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.type);
    s << indent << "color: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.color);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RR_PLUGIN_MESSAGE_ADDGROUPREQUEST_H
