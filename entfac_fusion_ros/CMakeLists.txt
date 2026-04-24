cmake_minimum_required(VERSION 3.0.2)
project(entfac_fusion_ros)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  message_filters
  entfac_fusion_core
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/ColoredPclTuning.cfg
)

catkin_package(
  CATKIN_DEPENDS
    dynamic_reconfigure
    geometry_msgs
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    std_srvs
    tf2_ros
    message_filters
    entfac_fusion_core
)

catkin_install_python(
  PROGRAMS
    scripts/colored_pcl_node.py
    scripts/pose_to_tf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/test_colored_pcl_node.test)
endif()
