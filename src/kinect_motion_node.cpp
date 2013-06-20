 
#include "KinectMotion.h" 
 
void
thread_function()
{
  while(ros::ok())
  {

  }
}


int
main(int argc, char** argv)
{
  ros::init(argc, argv, "my_image_reader");

  ros::NodeHandle color_frame_capture_handle;
  ros::NodeHandle depth_frame_capture_handle;
  ros::NodeHandle point_cloud_capture_handle;

  KinectMotion kinect_motion;

  cv::namedWindow(kinect_motion.COLOR_SCREEN);
  cv::namedWindow(kinect_motion.DEPTH_SCREEN);

  ros::Subscriber depth_frame_capture_subscriber = depth_frame_capture_handle.subscribe("camera/depth_registered/image_rect", 1,
                                                                                       &KinectMotion::depthImageCallback, &kinect_motion);
  ros::Subscriber color_frame_capture_subscriber = color_frame_capture_handle.subscribe("/camera/rgb/image_color", 1,
                                                                                       &KinectMotion::colorImageCallback, &kinect_motion);
  ros::Subscriber point_cloud_capture_subscriber = point_cloud_capture_handle.subscribe("/camera/depth_registered/points", 1,
                                                                                       &KinectMotion::pointCloudCallback, &kinect_motion);

  boost::thread thread(boost::bind(&KinectMotion::threadFunction, &kinect_motion));

  ROS_INFO("ROS is spinning ... \n\n");
  ros::spin();

  // Dont return from the main thread before Kinect::Motion::threadFunction returns.
  thread.join();

  return (0);
}
