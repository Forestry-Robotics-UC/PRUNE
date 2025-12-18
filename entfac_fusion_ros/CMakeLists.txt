cmake_minimum_required(VERSION 3.0.2)
project(entfac_fusion_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  tf2_ros
  message_filters
  entfac_fusion_core
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    tf2_ros
    message_filters
    entfac_fusion_core
)

catkin_install_python(
  PROGRAMS
    scripts/semantic_pcl_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/test_semantic_pcl_node.test)
endif()
