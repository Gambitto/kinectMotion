
#include "KinectMotion.h"

const std::string KinectMotion::COLOR_SCREEN = "COLOR_SCREEN";
const std::string KinectMotion::DEPTH_SCREEN = "DEPTH_SCREEN";
const std::string KinectMotion::MATCHES_SCREEN = "MATCHES_SCREEN";

void
KinectMotion::colorImageCallback (const sensor_msgs::ImageConstPtr& color_image_msg)
{

  cv_bridge::CvImageConstPtr bridge_image_ptr;

  try
  {
    bridge_image_ptr = cv_bridge::toCvCopy(color_image_msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
  }

  /* This is done for avoiding core dumps */
  this->mutex_.lock();
  this->color_image_ = bridge_image_ptr->image.clone();
  this->mutex_.unlock();

  cv::imshow(this->COLOR_SCREEN, this->color_image_);
  cv::waitKey(2);

}

void
KinectMotion::depthImageCallback (const sensor_msgs::ImageConstPtr& depth_image_msg)
{

  cv_bridge::CvImageConstPtr bridge_image_ptr;

  try
  {
    bridge_image_ptr = cv_bridge::toCvCopy(depth_image_msg);
  }
  catch(cv_bridge::Exception& e)
  {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
  }

  /* This is done for avoiding code dumps */
  this->mutex_.lock();
  this->depth_image_ = bridge_image_ptr->image.clone();
  this->mutex_.unlock();

  cv::imshow(this->DEPTH_SCREEN, this->depth_image_);
  cv::waitKey(2);

}

void
KinectMotion::pointCloudCallback (const sensor_msgs::PointCloud2& point_cloud_msg)
{
  /* This is for avoiding an intent of copy of point_cloud_ while this is being updated */
  this->mutex_.lock();
  pcl::fromROSMsg(point_cloud_msg, this->point_cloud_);
  this->mutex_.unlock();
}

void
KinectMotion::threadFunction()
{
  /* TODO This should be done in the constructor */
  this->viewer_.reset(new pcl::visualization::CloudViewer("CLOUD_VIEWER"));

  /* TODO Add here the state machine implementation */
  while(ros::ok())
  {
    if(this->get3DModel3 ())
    {
      ROS_INFO("Capture done! ");
    }
  }
}

std::vector<std::vector< cv::Point3f > >
KinectMotion::getValid3DCoordinates (std::vector<cv::Point2f> input1, cv::Mat &depth_frame_1, std::vector<cv::Point2f> input2, cv::Mat &depth_frame_2)
{
  /* TODO This values should be gotten from the CParameters file. The reading of this file should be done in the constructor */
  double fx = 520.841083;
  double fy = 519.896053;
  double cx = 320;
  double cy = 240;

  std::vector<cv::Point3f> output1;
  std::vector<cv::Point3f> output2;
  std::vector< std::vector<cv::Point3f > > out;

  for(int i=0; i<input1.size(); i++)
  {
    int u1 = input1[i].x;
    int v1 = input1[i].y;
    float z1 = depth_frame_1.at<float>(v1, u1);
    int u2 = input2[i].x;
    int v2 = input2[i].y;
    float z2 = depth_frame_2.at<float>(v2, u2);
    /* Check wether the depth pixels are virtuals(NaN) or not. */
    if(z1!=z1 || z2!=z2){continue;}
    float x1 = z1*(u1-cx)/fx;
    float y1 = z1*(v1-cy)/fy;
    cv::Point3f point1(x1, y1, z1);
    output1.push_back(point1);
    float x2 = z2*(u2-cx)/fx;
    float y2 = z2*(v2-cy)/fy;
    cv::Point3f point2(x2, y2, z2);
    output2.push_back(point2);
  }

  out.push_back(output1);
  out.push_back(output2);
  return (out);
}

std::vector<pcl::PointCloud<pcl::PointXYZ> >
KinectMotion::getValid3DKeypointsClouds(std::vector<cv::Point2f>& input_capture_2d_keypoints, cv::Mat& input_capture_depth_frame, std::vector<cv::Point2f>& local_capture_2d_keypoints, cv::Mat& local_capture_depth_frame)
{
  /* TODO This values should be gotten from the CParameters file. The reading of this file should be done in the constructor */
  double fx = 520.841083;
  double fy = 519.896053;
  double cx = 320;
  double cy = 240;

  pcl::PointCloud<pcl::PointXYZ> input_3d_keypoints;
  pcl::PointCloud<pcl::PointXYZ> local_3d_keypoints;

  std::vector<cv::Point2f>::iterator local_it = local_capture_2d_keypoints.begin();
  for(std::vector<cv::Point2f>::iterator input_it = input_capture_2d_keypoints.begin(); input_it != input_capture_2d_keypoints.end(); input_it++)
  {
    int u_input = (*input_it).x;
    int v_input = (*input_it).y;
    float z_input = input_capture_depth_frame.at<float>(v_input, u_input);
    int u_local = (*local_it).x;
    int v_local = (*local_it).y;
    float z_local = local_capture_depth_frame.at<float>(v_local, u_local);
    /* Check wether the depth pixels are virtuals(NaN) or not. */
    if(z_input!=z_input || z_local!=z_local){continue;}
    float x_input = z_input*(u_input-cx)/fx;
    float y_input = z_input*(v_input-cy)/fy;
    pcl::PointXYZ input_point(x_input, y_input, z_input);
    input_3d_keypoints.push_back(input_point);
    float x_local = z_local*(u_local-cx)/fx;
    float y_local = z_local*(v_local-cy)/fy;
    pcl::PointXYZ local_point(x_local, y_local, z_local);
    local_3d_keypoints.push_back(local_point);
    local_it++;
  }
    std::vector<pcl::PointCloud<pcl::PointXYZ> > return_value;
    return_value.push_back(input_3d_keypoints);
    return_value.push_back(local_3d_keypoints);
    return (return_value);
}

std::vector<pcl::PointCloud<pcl::PointXYZ> >
KinectMotion::findHighestLikelihoodMatch(Capture &capture, int &most_likely_matching_capture_index)
{

  std::vector<int> a_priori_valid_matches_number;
  std::vector<pcl::PointCloud<pcl::PointXYZ> > keypoints_clouds_vector;
  int highest_keypoints_hits = 0;
  std::vector<pcl::PointCloud<pcl::PointXYZ> > final_keypoints_clouds_vector;

  for(std::vector<Capture>::iterator it = this->captures_vector_.begin(); it != this->captures_vector_.end(); it++)
  {
    std::vector<cv::DMatch> matches;
    this->matcher_.match(capture.getDescriptors(), (*it).getDescriptors(), matches);

    /* Trying to improve the matching. */
    std::vector<cv::DMatch> good_matches;
    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < matches.size(); i++)
    {
      double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;

    }
    for (int i = 0; i < matches.size(); i++)
    {
        if( matches[i].distance < 2*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }

    /* Continue just if there are at least 10 good matches */
    if(good_matches.size() < 5)
    {
      most_likely_matching_capture_index = -1;
      this->mutex_.lock();
      std::cout << std::endl << "Not enough good matches" << std::endl;
      this->mutex_.unlock();
      keypoints_clouds_vector.clear();
      return keypoints_clouds_vector;
    }

    std::vector<cv::Point2f> matched_2d_input_capture_points;
    std::vector<cv::Point2f> matched_2d_local_capture_points;

    for(std::vector<cv::DMatch>::iterator matches_it = good_matches.begin(); matches_it != good_matches.end(); matches_it++)
    {
      matched_2d_input_capture_points.push_back(capture.getKeypoints()[(*matches_it).queryIdx].pt);
      matched_2d_local_capture_points.push_back((*it).getKeypoints()[(*matches_it).trainIdx].pt);
    }

    if(matched_2d_input_capture_points.size() != matched_2d_local_capture_points.size())
    {
        /* The output streams are shared objects, so the use between thread has to be coordinated */
        this->mutex_.lock();
        std::cerr << std::endl << "Error - KinectMotion::findHighestLikelihoodMatch = matched_2d_input_capture_points.size() != matched_2d_local_capture_points.size()" << std::endl;
        this->mutex_.unlock();
        most_likely_matching_capture_index = -1;
        keypoints_clouds_vector.clear();
        return (keypoints_clouds_vector);
    }

    keypoints_clouds_vector.clear();
    keypoints_clouds_vector = this->getValid3DKeypointsClouds(matched_2d_input_capture_points, capture.getDepthImage(), matched_2d_local_capture_points, (*it).getDepthImage());

    a_priori_valid_matches_number.push_back(keypoints_clouds_vector[0].size());
    this->mutex_.lock();
    std::cout << std::endl << " KinectMotion::findHighestLikelihoodMatch : There are " << keypoints_clouds_vector[0].size() << " 3d keypoints matches" << std::endl;
    this->mutex_.unlock();

    if(keypoints_clouds_vector[0].size() > highest_keypoints_hits)
    {
      highest_keypoints_hits = keypoints_clouds_vector[0].size();
      final_keypoints_clouds_vector = keypoints_clouds_vector;
    }

  }

  if(a_priori_valid_matches_number.size() == 0)
  {
      most_likely_matching_capture_index = -1;
      keypoints_clouds_vector.clear();
      this->mutex_.lock();
      std::cout << std::endl << " KinectMotion::findHighestLikelihoodMatch : No matches " << std::endl;
      this->mutex_.unlock();
      return keypoints_clouds_vector;
  }

  a_priori_valid_matches_number.push_back(1);
  std::vector<int>::iterator val = std::max_element(a_priori_valid_matches_number.begin(), a_priori_valid_matches_number.end());

  if( val == a_priori_valid_matches_number.end())
  {
    most_likely_matching_capture_index = -1;
    this->mutex_.lock();
    std::cout << std::endl << "Not enough 3d keypoints matches" << std::endl;
    this->mutex_.unlock();
    keypoints_clouds_vector.clear();
    return keypoints_clouds_vector;
  }
  most_likely_matching_capture_index = a_priori_valid_matches_number.size() - (a_priori_valid_matches_number.end() - val);

  return (final_keypoints_clouds_vector);
}

bool
KinectMotion::getRigidBodyTransformation(pcl::PointCloud<pcl::PointXYZ> source_keypoints_cloud, pcl::PointCloud<pcl::PointXYZ> target_keypoints_cloud, Eigen::Matrix4f & transformation)
{
  pcl::PointCloud<pcl::PointXYZ> source_cloud;
  source_cloud.width = source_keypoints_cloud.size();
  source_cloud.height = 1;
  source_cloud.is_dense = false;
  pcl::PointCloud<pcl::PointXYZ> target_cloud;
  target_cloud.width = target_keypoints_cloud.size();
  target_cloud.height = 1;
  target_cloud.is_dense = false;

  /* Adapt the input clouds to the requirements */
  for(int i = 0; i < source_keypoints_cloud.size(); i++)
  {
      source_cloud.push_back(source_keypoints_cloud.points[i]);
      target_cloud.push_back(target_keypoints_cloud.points[i]);
  }

  if(source_cloud.size() < 7)
  {
    this->mutex_.lock();
    std::cout << std::endl << "KinectMotion::getRigidBodyTransformation : Not enough points for filtering " << std::endl;
    this->mutex_.unlock();
    return (false);
  }

  ModalFilter::filter(source_cloud, target_cloud);

  this->icp_.setRANSACOutlierRejectionThreshold(0.0005);
  this->icp_.setEuclideanFitnessEpsilon(0.005);

  this->icp_.setInputCloud(source_cloud.makeShared());
  this->icp_.setInputTarget(target_cloud.makeShared());

  //this->icp_.setMaxCorrespondenceDistance(0.5);
  //this->icp_.setTransformationEpsilon(0.5);

  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  this->icp_.align(aligned_cloud);

  /* Results presentation*/
  this->mutex_.lock();
  std::cout << std::endl << "Has converged: " << this->icp_.hasConverged()<< " Score: " <<
                this->icp_.getFitnessScore() << std::endl;
  this->mutex_.unlock();

  if(!this->icp_.hasConverged())
  {
    this->mutex_.lock();
    std::cout << std::endl << "The ICP algorithm didn't converged." << std::endl;
    this->mutex_.unlock();
    return (false);
  }
  else if((this->icp_.getFitnessScore() > 0.05))
  {
    this->mutex_.lock();
    std::cout << std::endl << "The ICP fitness score is too high = " << this->icp_.getFitnessScore() << std::endl;
    this->mutex_.unlock();
    return (false);
  }

  Eigen::Matrix4f direct_transformation(this->icp_.getFinalTransformation());
  transformation = direct_transformation;
  return true;

}

bool
KinectMotion::get3DModel ()
{
  ROS_INFO("Press a key to make a capture. ");
  char a;
  std::cin >> a;

  /* Validate the captured data. */
  if(this->color_image_.data == NULL || this->depth_image_.data == NULL || this->point_cloud_.size() == 0)
  {
    ROS_INFO("Invalid data.");
    return false;
  }

  /* Generate the local copys of the frames and the cloud. */
  this->mutex_.lock();
  cv::Mat local_color_image = this->color_image_.clone();
  cv::Mat local_depth_image = this->depth_image_.clone();
  pcl::PointCloud<pcl::PointXYZRGB> local_cloud;
  pcl::copyPointCloud(this->point_cloud_, local_cloud);
  this->mutex_.unlock();

  /* Find interest points in the color frame. What detector to use? SURF. */
  /* Need to define the detector, the descriptor and the matcher. */
  std::vector<cv::KeyPoint> local_image_keypoints;
  cv::Mat local_image_descriptors;
  detector_.detect(local_color_image, local_image_keypoints);
  extractor_.compute(local_color_image, local_image_keypoints, local_image_descriptors);

  /* Create and store the last capture */
  /* Initialization of the capture's corresponding rigid body transformation */
  Eigen::Matrix4f transformation;
  transformation.setZero();
  transformation << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
  Capture capture(local_color_image, local_depth_image, local_cloud, local_image_keypoints, local_image_descriptors, transformation);
  this->captures_vector_.push_back(capture);

  if(this->captures_vector_.size() == 1)
  {
    this->complete_cloud_ += capture.cloud_;
  }
  else if(this->captures_vector_.size() == 2)
  {
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> good_matches;
    cv::Mat output_image;

    this->matcher_.match(this->captures_vector_[0].descriptors_, this->captures_vector_[1].descriptors_, matches);

    /* Trying to improve the matching. */
    double max_dist = 0; double min_dist = 100;
    for (int i=0; i < this->captures_vector_[0].descriptors_.rows; i++)
    {
      double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;

    }
    for (int i = 0; i < captures_vector_[0].descriptors_.rows; i++)
    {
        if( matches[i].distance < 2*min_dist )
        {
            good_matches.push_back( matches[i]);
            //std::cout << "\nMatch " << i << " = " << matches[i].distance << std::endl;
        }
    }
    std::cout << std::endl << "There are " << good_matches.size() << " good matches" << std::endl;

    cv::drawMatches(this->captures_vector_[0].color_image_, this->captures_vector_[0].keypoints_, this->captures_vector_[1].color_image_, this->captures_vector_[1].keypoints_, good_matches, output_image, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow(this->MATCHES_SCREEN, output_image);


    if(good_matches.size() > 5)

    {
      std::vector<cv::Point2f> matched_points_1;
      std::vector<cv::Point2f> matched_points_2;
      std::vector<cv::Point3f> matched_3d_points_1;
      std::vector<cv::Point3f> matched_3d_points_2;
      for(int i = 0; i < good_matches.size(); i++)
      {
        matched_points_1.push_back(this->captures_vector_[0].keypoints_[good_matches[i].queryIdx].pt);
        matched_points_2.push_back(this->captures_vector_[1].keypoints_[good_matches[i].trainIdx].pt);
      }
      std::vector<std::vector< cv::Point3f > > matched_3d_points;
      matched_3d_points = getValid3DCoordinates(matched_points_1, this->captures_vector_[0].depth_image_, matched_points_2, this->captures_vector_[1].depth_image_);
      matched_3d_points_1 = matched_3d_points[0];
      matched_3d_points_2 = matched_3d_points[1];
      //std::cout << std::endl << "Matched input 3d points: " << matched_3d_points_1 << std::endl << "Matched output 3d points: " << matched_3d_points_2 <<std::endl;

      pcl::PointCloud<pcl::PointXYZ> cloud11;
      pcl::PointCloud<pcl::PointXYZ> cloud22;
      pcl::PointCloud<pcl::PointXYZ> finalCloud2;
      cloud11.width = matched_3d_points_1.size();
      cloud11.height = 1;
      cloud11.is_dense = false;
      cloud11.points.resize(cloud11.width*cloud11.height);
      cloud22.width = matched_3d_points_2.size();
      cloud22.height = 1;
      cloud22.is_dense = false;
      cloud22.points.resize(cloud22.width*cloud22.height);

      for(int i = 0; i < matched_3d_points_1.size(); i++)
      {
          cloud11.points[i].x = matched_3d_points_1[i].x;
          cloud11.points[i].y = matched_3d_points_1[i].y;
          cloud11.points[i].z = matched_3d_points_1[i].z;
          cloud22.points[i].x = matched_3d_points_2[i].x;
          cloud22.points[i].y = matched_3d_points_2[i].y;
          cloud22.points[i].z = matched_3d_points_2[i].z;
      }

      ModalFilter::filter(cloud11, cloud22);

      this->icp_.setInputCloud(cloud11.makeShared());
      this->icp_.setInputTarget(cloud22.makeShared());
      this->icp_.setRANSACOutlierRejectionThreshold(0.05);
      //this->icp_.setMaxCorrespondenceDistance(0.09);
      this->icp_.align(finalCloud2);
      std::cout << "Has converged: " << this->icp_.hasConverged()<< " score: " <<
                    this->icp_.getFitnessScore() << std::endl;
      std::cout << this->icp_.getFinalTransformation() << std::endl << cloud22.size()<< std::endl;
      if(this->icp_.hasConverged() && this->icp_.getFitnessScore() < 0.05)
      {

        pcl::PointCloud<pcl::PointXYZRGB> show;
        Eigen::Matrix4f trans(this->icp_.getFinalTransformation());
        Eigen::Matrix4f trans2(trans.inverse());
        pcl::transformPointCloud(this->captures_vector_[1].cloud_, show, trans2);
        //this->complete_cloud_ += show;
        show += this->captures_vector_[0].cloud_;

        this->viewer_->showCloud(show.makeShared());
      }

      //std::cout << std::endl << "The transformation: " <<std::endl<< this->transformation_estimator_.getTransformation().matrix() << std::endl;
    }

    this->captures_vector_.erase(this->captures_vector_.begin());
  }

}


bool
KinectMotion::get3DModel2 ()
{
  ROS_INFO("Press  a key to make a capture. ");
  char a;
  std::cin >> a;

  /* Validate the captured data. */
  if(this->color_image_.data == NULL || this->depth_image_.data == NULL || this->point_cloud_.size() == 0)
  {
    ROS_INFO("Invalid data.");
    return false;
  }

  /* Generate the local copys of the frames and the cloud. */
  cv::Mat local_color_image = this->color_image_.clone();
  cv::Mat local_depth_image = this->depth_image_.clone();
  pcl::PointCloud<pcl::PointXYZRGB> local_cloud;
  pcl::copyPointCloud(this->point_cloud_, local_cloud);

  std::vector<cv::KeyPoint> local_image_keypoints;
  cv::Mat local_image_descriptors;

  /* Create and store the last capture */
  /* Initialization of the capture's corresponding rigid body transformation */
  Eigen::Matrix4f transformation;
  transformation.setZero();
  transformation << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
  Capture capture(local_color_image, local_depth_image, local_cloud, local_image_keypoints, local_image_descriptors, transformation);
  this->captures_vector_.push_back(capture);

  if(this->captures_vector_.size() == 1)
  {
    this->complete_cloud_ += capture.cloud_;
  }
  else if(this->captures_vector_.size() == 2)
  {

    pcl::PointCloud<pcl::PointXYZ> cloud1;
    pcl::PointCloud<pcl::PointXYZ> cloud2;
    cloud1.width = 640*480;
    cloud1.height = 1;
    cloud1.is_dense = false;
    cloud1.points.resize(cloud1.width*cloud1.height);
    cloud2.width = 640*480;
    cloud2.height = 1;
    cloud2.is_dense = false;
    cloud2.points.resize(cloud2.width*cloud2.height);

    pcl::PointCloud<pcl::PointXYZ> final_cloud;

    for (int i = 0; i < this->captures_vector_[0].cloud_.points.size(); i++)
    {
      pcl::PointXYZ point1;
      pcl::PointXYZ point2;
      point1.x = this->captures_vector_[0].cloud_.points[i].x;
      point1.y = this->captures_vector_[0].cloud_.points[i].y;
      point1.z = this->captures_vector_[0].cloud_.points[i].z;
      point2.x = this->captures_vector_[1].cloud_.points[i].x;
      point2.y = this->captures_vector_[1].cloud_.points[i].y;
      point2.z = this->captures_vector_[1].cloud_.points[i].z;
      cloud1.push_back(point1);
      cloud2.push_back(point2);
    }
    std::cout << cloud1.size()<< "  " << cloud2.size() << std::endl;

    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setLeafSize(0.1, 0.1, 0.1);
    filter.setInputCloud(cloud1.makeShared());
    //pcl::PointCloud<pcl::PointXYZ> cloud1_filtered;
    filter.filter(cloud1);
    filter.setInputCloud(cloud2.makeShared());
    filter.filter(cloud2);
    std::cout << "\n Finished filtering \n";

    this->icp_.setInputCloud(cloud2.makeShared());
    this->icp_.setInputTarget(cloud1.makeShared());
    this->icp_.align(final_cloud);
    std::cout << "Has converged: " << this->icp_.hasConverged()<< " score: " <<
                 this->icp_.getFitnessScore() << std::endl;
    std::cout << this->icp_.getFinalTransformation() << std::endl;

    this->captures_vector_.erase(this->captures_vector_.end());
  }

}


bool
KinectMotion::get3DModel3 ()
{
  ROS_INFO("Press a key to make a capture. ");
  char a;
  std::cin >> a;

  /* Validate the captured data. */
  if(this->color_image_.data == NULL || this->depth_image_.data == NULL || this->point_cloud_.size() == 0)
  {
    ROS_INFO("Invalid data.");
    return false;
  }

  /* Generate the local copys of the frames and the cloud. */
  this->mutex_.lock();
  cv::Mat local_color_image = this->color_image_.clone();
  cv::Mat local_depth_image = this->depth_image_.clone();
  pcl::PointCloud<pcl::PointXYZRGB> local_cloud;
  pcl::copyPointCloud(this->point_cloud_, local_cloud);
  this->mutex_.unlock();

  /* Initialization of the capture's corresponding rigid body transformation */
  Eigen::Matrix4f transformation;
  transformation.setZero();
  transformation << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

  std::vector<cv::KeyPoint> local_image_keypoints;
  cv::Mat local_image_descriptors;
  Capture capture(local_color_image, local_depth_image, local_cloud, local_image_keypoints, local_image_descriptors, transformation);
  this->detector_.detect(capture.getColorImage(), capture.getKeypoints());
  this->extractor_.compute(capture.getColorImage(), capture.getKeypoints(), capture.getDescriptors());

  if(this->captures_vector_.size() == 0)
  {
    this->captures_vector_.push_back(capture);
  }
  else
  {
    int matching_capture_index = -1;
    std::vector<pcl::PointCloud<pcl::PointXYZ> > keypoints_clouds;
    keypoints_clouds = this->findHighestLikelihoodMatch(capture, matching_capture_index);
    if(matching_capture_index == -1)
    {
        this->mutex_.lock();
        std::cout << std::endl << "KinectMotion::get3dModel() - No matching capture found " << std::endl;
        this->mutex_.unlock();
        return (false);
    }

    this->mutex_.lock();
    std::cout << std::endl << "KinectMotion::get3dModel() : The most likely match will be with capture number: "<< matching_capture_index << std::endl;
    if(keypoints_clouds.size() == 0)
    {
      std::cout << std::endl << "KinectMotion::get3dModel() : keypoints_clouds vector is empty " << std::endl;
      return (false);
    }
    std::cout << std::endl << "KinectMotion::get3dModel() : The Corresponding number of keypoints: "<< keypoints_clouds[0].size() << " - " << keypoints_clouds[1].size() << std::endl;
    this->mutex_.unlock();

    Eigen::Matrix4f transformation;
    if(!this->getRigidBodyTransformation(keypoints_clouds[0], keypoints_clouds[1], transformation))
    {
      this->mutex_.lock();
      std::cout << std::endl << "KinectMotion::get3dModel() : Unable to get rigid body transformation " << std::endl;
      this->mutex_.unlock();
      return (false);
    }

    Eigen::Matrix4f final_transformation(this->captures_vector_[matching_capture_index].getTransformation()*transformation);
    capture.setTransformation(final_transformation);
    this->captures_vector_.push_back(capture);

    this->mutex_.lock();
    std::cout << std::endl << "KinectMotion::get3dModel() : The transformation: " << std::endl << final_transformation<< std::endl;
    this->mutex_.unlock();

    this->complete_cloud_.clear();
    for(int i = 0; i < this->captures_vector_.size(); i++)
    {
      pcl::PointCloud<pcl::PointXYZRGB> temp_cloud;
      Eigen::Matrix4f tr(this->captures_vector_[i].getTransformation());
      pcl::transformPointCloud(this->captures_vector_[i].cloud_, temp_cloud, tr);
      this->complete_cloud_ += temp_cloud;
    }

    pcl::VoxelGrid<pcl::PointXYZRGB> filter;
    filter.setLeafSize(0.005, 0.005, 0.005);
    filter.setInputCloud(this->complete_cloud_.makeShared());
    //pcl::PointCloud<pcl::PointXYZ> cloud1_filtered;
    filter.filter(this->complete_cloud_);

    this->viewer_->showCloud(this->complete_cloud_.makeShared());

    return (true);

  }
}
