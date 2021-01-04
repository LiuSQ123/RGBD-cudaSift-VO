//
// Created by liushiqi on 19-8-17.
//
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>

#include "featureTracking.h"

using namespace std;

int main(int argc, char **argv)
{
    VO::featureTracking vo;
    cv::Mat image0 = cv::imread("/home/liushiqi/ClionProjects/CUDA_project1/data/color/1.png",cv::IMREAD_GRAYSCALE);
    cv::Mat depth0 = cv::imread("/home/liushiqi/ClionProjects/CUDA_project1/data/depth/1.png");

    cv::Mat image1 = cv::imread("/home/liushiqi/ClionProjects/CUDA_project1/data/color/2.png",cv::IMREAD_GRAYSCALE);
    cv::Mat depth1 = cv::imread("/home/liushiqi/ClionProjects/CUDA_project1/data/depth/2.png");

    cv::Mat image2 = cv::imread("/home/liushiqi/ClionProjects/CUDA_project1/data/color/3.png",cv::IMREAD_GRAYSCALE);
    cv::Mat depth2 = cv::imread("/home/liushiqi/ClionProjects/CUDA_project1/data/depth/3.png");

    VO::featureTracking::cam = Eigen::Matrix3d::Zero();
    VO::featureTracking::cam(0,0)=525.0;
    VO::featureTracking::cam(1,1)=525.0;
    VO::featureTracking::cam(0,2)=319.5;
    VO::featureTracking::cam(1,2)=239.5;
    VO::featureTracking::cam(2,2)=1.0;
    BA::costFunction_reprojection::cam = VO::featureTracking::cam;
    vo.extractFeature(image0,depth0);
    vo.extractFeature(image1,depth1);
    vo.optimization();
    vo.show();
    vo.extractFeature(image2,depth2);
    vo.optimization();
    vo.show();
    //vo.slidWindow();
    cv::imshow("1",image2);
    cv::waitKey(0);
    return 0;
}
