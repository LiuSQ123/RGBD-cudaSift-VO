//
// Created by liushiqi on 19-7-31.
//

#include "VO.h"
using namespace VO;
using namespace std;
visualOdometry::visualOdometry(Senser _type, unsigned long windowSize){
    /*
    pose.resize(windowSize,Eigen::Matrix4d::Identity());
    VOSenser = _type;
    VOState = UNINIT;
    featureTrackingPtr = featureTracking::ptr(new featureTracking);
    frameDataPtr = frameData::ptr(new frameData(50));
    featureTrackingPtr->pframeData = frameDataPtr;
     */
}
bool visualOdometry::VOinit() {
    cout<<"VO init Finish!"<<endl;
}
void visualOdometry::RGBDTracking(cv::Mat image,cv::Mat depth){
    /*
    /// 针对于RGBD相机,不需要初始化
    /// 首先提取特征点
    featureTrackingPtr->RGBDTracking(image,depth);
    /// 然后进行BA优化
    optimization();
    /// 判断是否属于关键帧,然后在其中进行滑动窗口更新
    judgeAndInsertKeyFrame();
    */
}