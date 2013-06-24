 
#ifndef KINECT_MOTION_H_
#define KINECT_MOTION_H_

#include <iostream>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <sensor_msgs/image_encodings.h>
#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/centroid.h>
#include <Eigen/Dense>

#include <boost/thread.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <math.h>

class ModalFilter
{
public:
  static bool
  calculateDecoupledDistanceClouds(const pcl::PointCloud<pcl::PointXYZ>& input_cloud, std::vector<pcl::PointCloud<pcl::PointXYZ> >& output_clouds)
  {
    if(input_cloud.empty())
    {
        std::cout << std::endl << "ModalFilter::calculateDecoupledDistanceClouds ERROR : input cloud is empty " << std::endl;
        return false;
    }

    output_clouds.clear();

    for(pcl::PointCloud<pcl::PointXYZ>::const_iterator outter_it = input_cloud.begin(); outter_it != input_cloud.end(); outter_it++)
    {
      pcl::PointCloud<pcl::PointXYZ> inner_cloud;

      for(pcl::PointCloud<pcl::PointXYZ>::const_iterator inner_it = input_cloud.begin(); inner_it != input_cloud.end(); inner_it++)
      {
        pcl::PointXYZ distance;

        if(inner_it != outter_it)
        {
          distance.x = outter_it->x - inner_it->x;
          distance.y = outter_it->y - inner_it->y;
          distance.z = outter_it->z - inner_it->z;
          inner_cloud.push_back(distance);
        }

      }

      output_clouds.push_back(inner_cloud);
    }

    return true;
  }


  /* The use of this function should be protected by a mutex. */
  static bool
  filter(pcl::PointCloud<pcl::PointXYZ>& source_cloud, pcl::PointCloud<pcl::PointXYZ>& target_cloud)
  {
    /* The two input clouds must have the same size */
    if(source_cloud.size() != target_cloud.size())
    {
      std::cerr << "ModalFilter::filter Error = The input clouds have a different size";
      return false;
    }

    std::cout << std::endl << "The source and target clouds have " << source_cloud.size() << " elements." << std::endl;

    /* vectors containing the distance between each point and the other cloud's elements */
    std::vector<pcl::PointCloud<pcl::PointXYZ> > source_distance;
    std::vector<pcl::PointCloud<pcl::PointXYZ> > target_distance;

    /* Calculate source_distance. Find the distance vector for the source_cloud's elements */
    for(int i = 0; i < source_cloud.size(); i++)
    {
      pcl::PointCloud<pcl::PointXYZ> distance_inner_cloud;
      for(int j = 0; j < source_cloud.size(); j++)
      {
        pcl::PointXYZ distance;
        if(j != i )
        {
          distance.x = source_cloud.points[j].x - source_cloud.points[i].x;
          distance.y = source_cloud.points[j].y - source_cloud.points[i].y;
          distance.z = source_cloud.points[j].z - source_cloud.points[i].z;
          distance_inner_cloud.push_back(distance);
        }
      }
      source_distance.push_back(distance_inner_cloud);
    }

    // Calculate target_distance. Find the distance vector for the target_cloud's elements
    for(int i = 0; i < target_cloud.size(); i++)
    {
      pcl::PointCloud<pcl::PointXYZ> distance_inner_cloud;
      for(int j = 0; j < target_cloud.size(); j++)
      {
        pcl::PointXYZ distance;
        if(j != i )
        {
          distance.x = target_cloud.points[j].x - target_cloud.points[i].x;
          distance.y = target_cloud.points[j].y - target_cloud.points[i].y;
          distance.z = target_cloud.points[j].z - target_cloud.points[i].z;
          distance_inner_cloud.push_back(distance);
        }
      }
      target_distance.push_back(distance_inner_cloud);
    }

    //vector containing the error between the source and target distance vectors
    std::vector<pcl::PointCloud<pcl::PointXYZ> > error_distance_clouds;

    // find the "magnitud" error between the source and target distance vectors
    for(int i = 0; i < source_distance.size(); i++)
    {
      pcl::PointCloud<pcl::PointXYZ> error_inner_cloud;
      for(int j = 0; j < source_distance[i].size() ; j++)
      {
        pcl::PointXYZ error;
        error.x = (source_distance[i]).points[j].x - (target_distance[i]).points[j].x;
        error.y = (source_distance[i]).points[j].y - (target_distance[i]).points[j].y;
        error.z = (source_distance[i]).points[j].z - (target_distance[i]).points[j].z;
        error_inner_cloud.push_back(error);
      }
      error_distance_clouds.push_back(error_inner_cloud);
    }

    // time to remove the outliers :)
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outliers_filter;
    std::vector<pcl::PointCloud<pcl::PointXYZ> > inliers_clouds;
    std::vector<std::vector<int> > inliers_indices;

    std::cout << std::endl << "Results of the magnitud based filtering." << std::endl;

    for(int i = 0; i < error_distance_clouds.size(); i++)
    {
        outliers_filter.setMeanK(error_distance_clouds[i].size());
        outliers_filter.setStddevMulThresh(0.2);
        outliers_filter.setInputCloud(error_distance_clouds[i].makeShared());
        pcl::PointCloud<pcl::PointXYZ> inliers;
        std::vector<int> indices;
        outliers_filter.filter(inliers);
        outliers_filter.filter(indices);
        inliers_clouds.push_back(inliers);
        inliers_indices.push_back(indices);

        //std::cout << std::endl << "For the point " << i << " there are " << inliers.size() << " positive hits.\n";
        //for(int j = 0; j < indices.size(); j++){ std::cout <<indices[j]<<"-"; }
        /*for (size_t j = 0; j < inliers.points.size (); ++j)
          std::cerr << "    " << inliers.points[j].x << " "
                              << inliers.points[j].y << " "
                              << inliers.points[j].z << std::endl;*/
    }

    // Calculate the Mode
    std::vector<int> inliers_clouds_size;
    for(int i = 0; i < inliers_clouds.size(); i++)
    {
        inliers_clouds_size.push_back(inliers_clouds[i].size());
    }
    std::vector<int> histogram(100);
    for(int i = 0; i < inliers_clouds_size.size(); ++i)
    {
        histogram[inliers_clouds_size[i]]++;
    }

    int mode = histogram.size() - (int)(histogram.end() - std::max_element(histogram.begin(), histogram.end()));
    std::cout << "\nThe Mode is: " << mode << std::endl;

    // After pre-filtering we get:
    for(int i = 0; i < inliers_indices.size(); i++)
    {
        //std::cout << i;
        if(inliers_indices[i].size() != mode)
        {
            inliers_indices.erase(inliers_indices.begin() + i);
        }
    }

    //std::cout << "\nAfter pre-Filtering: The contents of inliers_indices\n";
    std::vector<int> inliers_histogram(100);
    for(int i = 0; i < inliers_indices.size(); i++)
    {
        //std::cout << std::endl;
        for(int j = 0; j < inliers_indices[i].size(); j++)
        {
            //std::cout << (inliers_indices[i])[j] << "-";
            inliers_histogram[(inliers_indices[i])[j]]++;
        }
    }


    pcl::PointCloud<pcl::PointXYZ> source_return_cloud;
    pcl::PointCloud<pcl::PointXYZ> target_return_cloud;
    //std::cout << std::endl << "The valid indices are: " << std::endl;
    for(int i = 0; i < inliers_histogram.size(); i++)
    {
        if(inliers_histogram[i] == inliers_indices.size())
        {
            //std::cout << "-" << i  ;
            source_return_cloud.push_back(source_cloud.points[i]);
            target_return_cloud.push_back(target_cloud.points[i]);
        }
    }

    source_cloud = source_return_cloud;
    target_cloud = target_return_cloud;





    //compareAngles(source_distance, target_distance);

  }

  static bool
  compareAngles(std::vector<pcl::PointCloud<pcl::PointXYZ> >& source_vectors_distance,
                std::vector<pcl::PointCloud<pcl::PointXYZ> >& target_vectors_distance)
  {

    // the two input clouds must have the same size
    if(source_vectors_distance.size() != target_vectors_distance.size())
    {
      std::cerr << "ModalFilter::compareAngles Error = The input vectors have a different size";
      return false;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZ> > source_vectors_angles;
    std::vector<pcl::PointCloud<pcl::PointXYZ> > target_vectors_angles;
    // The angles would be defined as: alpha, beta, and gamma
    // alpha = atan(y/x)
    // beta = atan(z/x)
    // gamma = atan(y/z)

    // calculation of the source vectors' angles
    std::cout << std::endl << "The source and target vector have: " << source_vectors_distance.size() << target_vectors_distance.size() <<std::endl;

    for(int i = 0; i < source_vectors_distance.size(); i++)
    {
      pcl::PointCloud<pcl::PointXYZ> source_angles_inner_cloud;
      pcl::PointCloud<pcl::PointXYZ> target_angles_inner_cloud;
      for(int j = 0; j < (source_vectors_distance[i].size()); j++)
      {
        pcl::PointXYZ source_angle;
        source_angle.x = atan((source_vectors_distance[i]).points[j].y/(source_vectors_distance[i]).points[j].x);
        source_angle.y = atan((source_vectors_distance[i]).points[j].z/(source_vectors_distance[i]).points[j].x);
        source_angle.z = atan((source_vectors_distance[i]).points[j].y/(source_vectors_distance[i]).points[j].z);
        pcl::PointXYZ target_angle;
        target_angle.x = atan((target_vectors_distance[i]).points[j].y/(target_vectors_distance[i]).points[j].x);
        target_angle.y = atan((target_vectors_distance[i]).points[j].z/(target_vectors_distance[i]).points[j].x);
        target_angle.z = atan((target_vectors_distance[i]).points[j].y/(target_vectors_distance[i]).points[j].z);
        source_angles_inner_cloud.push_back(source_angle);
        target_angles_inner_cloud.push_back(target_angle);
      }
      source_vectors_angles.push_back(source_angles_inner_cloud);
      target_vectors_angles.push_back(target_angles_inner_cloud);
    }

    std::vector<pcl::PointCloud<pcl::PointXYZ> > error_angles_clouds;

    // find the error between the source and target distance vectors
    for(int i = 0; i < source_vectors_angles.size(); i++)
    {

      pcl::PointCloud<pcl::PointXYZ> error_inner_cloud;
      for(int j = 0; j < (source_vectors_angles[i].size()); j++)
      {
        pcl::PointXYZ angle_error;
        angle_error.x = (source_vectors_angles[i]).points[j].x - (target_vectors_angles[i]).points[j].x;
        angle_error.y = (source_vectors_angles[i]).points[j].y - (target_vectors_angles[i]).points[j].y;
        angle_error.z = (source_vectors_angles[i]).points[j].z - (target_vectors_angles[i]).points[j].z;
        error_inner_cloud.push_back(angle_error);
      }
      error_angles_clouds.push_back(error_inner_cloud);
    }

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outliers_filter;
    std::vector<pcl::PointCloud<pcl::PointXYZ> > inliers_clouds;
    std::vector<std::vector<int> > inliers_indices;

    std::cout << std::endl << "Results of the orientation based filtering." << std::endl;

    for(int i = 0; i < error_angles_clouds.size(); i++)
    {
        outliers_filter.setMeanK(error_angles_clouds[i].size());
        outliers_filter.setStddevMulThresh(0.2);
        outliers_filter.setInputCloud(error_angles_clouds[i].makeShared());
        pcl::PointCloud<pcl::PointXYZ> inliers;
        std::vector<int> indices;
        outliers_filter.filter(inliers);
        outliers_filter.filter(indices);
        inliers_clouds.push_back(inliers);
        inliers_indices.push_back(indices);

        std::cout << std::endl << "For the point " << i << " there are " << inliers.size() << " positive hits.\n";
        for(int j = 0; j < indices.size(); j++){ std::cout <<indices[j]<<"-"; }
        /*for (size_t j = 0; j < inliers.points.size (); ++j)
          std::cerr << "    " << inliers.points[j].x << " "
                              << inliers.points[j].y << " "
                              << inliers.points[j].z << std::endl;*/
    }

    // Calculate the Mode
    std::vector<int> inliers_clouds_size;
    for(int i = 0; i < inliers_clouds.size(); i++)
    {
        inliers_clouds_size.push_back(inliers_clouds[i].size());
    }
    std::vector<int> histogram(100);
    for(int i = 0; i < inliers_clouds_size.size(); ++i)
    {
        histogram[inliers_clouds_size[i]]++;
    }
    std::cout << "\nThe Mode is: " <<histogram.size() - (int)(histogram.end() - std::max_element(histogram.begin(), histogram.end())) << std::endl;
  }


};

class Capture{
public:
  cv::Mat color_image_;
  cv::Mat depth_image_;
  pcl::PointCloud<pcl::PointXYZRGB> cloud_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
  Eigen::Matrix4f transformation_;
  Capture(cv::Mat color, cv::Mat depth, pcl::PointCloud<pcl::PointXYZRGB> cloud, std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors, Eigen::Matrix4f transformation)
  {
    this->color_image_ = color;
    this->depth_image_ = depth;
    this->cloud_ = cloud;
    this->keypoints_ = keypoints;
    this->descriptors_ = descriptors;
    this->transformation_ = transformation;
  }
  cv::Mat&
  getColorImage(){ return (this->color_image_);}
  cv::Mat&
  getDepthImage(){ return (this->depth_image_);}
  std::vector<cv::KeyPoint>&
  getKeypoints(){ return (this->keypoints_);}
  cv::Mat&
  getDescriptors(){ return (this->descriptors_);}
  Eigen::Matrix4f
  getTransformation(){ return (this->transformation_); }
  void
  setTransformation(Eigen::Matrix4f& transformation){ this->transformation_ = transformation;}
};

class KinectMotion
{
public:
  static const std::string COLOR_SCREEN;
  static const std::string DEPTH_SCREEN;
  static const std::string MATCHES_SCREEN;

  void
  colorImageCallback (const sensor_msgs::ImageConstPtr& color_image_msg);
  void
  depthImageCallback (const sensor_msgs::ImageConstPtr& depth_image_msg);
  void
  pointCloudCallback (const sensor_msgs::PointCloud2& point_cloud_msg);
  bool
  get3DModel ();
  bool
  get3DModel2 ();
  bool
  get3DModel3 ();
  std::vector<pcl::PointCloud<pcl::PointXYZ> >
  findHighestLikelihoodMatch(Capture& capture, int& most_likely_matching_capture_index);
  std::vector<pcl::PointCloud<pcl::PointXYZ> >
  getValid3DKeypointsClouds(std::vector<cv::Point2f>& input_capture_2d_keypoints, cv::Mat& input_capture_depth_frame, std::vector<cv::Point2f>& local_capture_2d_keypoints, cv::Mat& local_capture_depth_frame);
  bool
  getRigidBodyTransformation(pcl::PointCloud<pcl::PointXYZ> source_keypoints_cloud, pcl::PointCloud<pcl::PointXYZ> target_keypoints_cloud, Eigen::Matrix4f& transformation);
  void
  threadFunction ();
  std::vector<std::vector< cv::Point3f > >
  getValid3DCoordinates (std::vector<cv::Point2f> input1, cv::Mat &depth_frame_1, std::vector<cv::Point2f> input2, cv::Mat &depth_frame_2);

  //KinectMotion(){ this->matcher_ =  new cv::BFMatcher(cv::NORM_L2SQR, true); ROS_INFO("Constructor called.");}
  //~KinectMotion(){ delete this->matcher_;}

private:
  /* Data containers */
  cv::Mat color_image_;
  cv::Mat depth_image_;
  pcl::PointCloud<pcl::PointXYZRGB> point_cloud_;
  pcl::PointCloud<pcl::PointXYZRGB> complete_cloud_;
  std::vector<Capture> captures_vector_;

  /* Features detector, descriptor extractor and matcher */
  cv::SurfFeatureDetector detector_;
  cv::SurfDescriptorExtractor extractor_;
  cv::FlannBasedMatcher matcher_;

  /* Visualization */
  boost::shared_ptr<pcl::visualization::CloudViewer> viewer_;

  /* Rigid body transformation */
  pcl::TransformationFromCorrespondences transformation_estimator_;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;

  /* Access synchronization */
  boost::mutex mutex_;

};




#endif
