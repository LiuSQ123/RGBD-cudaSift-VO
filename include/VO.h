//
// Created by liushiqi on 19-7-31.
//

#ifndef FEATURETRACKING_VO_H
#define FEATURETRACKING_VO_H

#include "featureTracking.h"
#include <iostream>
using namespace std;
namespace VO{
    enum State {LOST, UNINIT, OK , RELOCATION};
    enum Senser {RGBD, MONO, STEREO};
    class visualOdometry{
    public:
        /// 默认为RGBD 相机
        visualOdometry(Senser _type = RGBD,unsigned long windowSize = 10);
        ~visualOdometry();
        bool VOinit();
        /// 深度视觉追踪
        void RGBDTracking(cv::Mat image,cv::Mat depth);
        /// 双目视觉追踪
        void stereoTracking(cv::Mat imageLeft,cv::Mat imageRight);
        /// 判断是否关键帧并插入
        void judgeAndInsertKeyFrame();
        /// 返回里程计状态
        State state();
        /// 丢失后重定位
        bool relocation();
        /// 获取当前最新状态
        Eigen::Matrix4d getPose();

    private:
        /// BA优化
        void optimization();
        /// 删除滑动窗口内最新帧
        void slidWindowNew();
        /// 删除
        void slidWindowOld();
    private:
        State VOState;
        Senser VOSenser;
        vector<Eigen::Matrix4d> pose;
        featureTracking::ptr featureTrackingPtr;
        //frameData::ptr frameDataPtr;
    };
}
#endif //FEATURETRACKING_VO_H
