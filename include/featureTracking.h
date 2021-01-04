//
// Created by liushiqi on 19-7-31.
//

#ifndef FEATURETRACKING_FEATURETRACKING_H
#define FEATURETRACKING_FEATURETRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <deque>
#include <unordered_map>
#include <unordered_set>

#include "cudaImage.h"
#include "cudaSift.h"

#include "frame.h"
#include "feature.h"
#include "visualization.h"

#include "problem.h"
#include "costFunction.h"
using namespace std;
namespace VO{
    /// 封装的cuda Sift特征
    class cuSiftData{

    };
    class keyFrame;
    /// 特征点库
    class featureData{
        //unordered_map<int,feature::ptr>;
        //unordered_set<feature::ptr> data;
    };
    /// 关键帧库
    /*
    class frameData{
    public:
        typedef std::shared_ptr<frameData> ptr;
        frameData(int _size):size(_size){}
        frameData() = delete;
        ~frameData() = default;

        /// 添加关键帧
        void addKeyFrame();
        /// 删除关键帧
        void eraseKeyFrame();
        /// 更新特征id
        void updateID();

        keyFrame::ptr getLastKeyFrame();

    private:
        int size;
        std::vector<keyFrame::ptr> keyFrames;
    };
    */
    class featureDataBase{
    public:
        unordered_map<int,feature::ptr> Data;
    };


    class featureTracking{
    public:
        typedef std::shared_ptr<featureTracking> ptr;
        typedef Eigen::Matrix<double,6,1> pose;
        typedef std::shared_ptr<pose> posePtr;
        featureTracking(int _databaseSize = 50);
        //featureTracking() = delete;
        ~featureTracking() = default;
        /// 深度视觉追踪
        bool RGBDTracking(cv::Mat image,cv::Mat depth,double time = 0.0);
        /// 双目视觉追踪
        void stereoTracking(cv::Mat imageLeft,cv::Mat imageRight);
        /// 整体优化
        bool optimization();
        /// 获取最新一帧数据
        keyFrame getLastFrame();
        /// 获取当前滑动窗口内关键帧的数目
        int getKeyFrameCount();
        /// 丢失后重新定位
        bool relocation();
        /// 滑动后更新ID
        void updateID();
        /// 获取当前帧特征提取的数量
        int getCurrentPointNum();
        bool extractFeature(cv::Mat image,cv::Mat depth = cv::Mat());
        /// 滑动窗口
        void slidWindow();

        void resetAllData();
        void clearAllData();
        /// 释放GPU显存
        void FreeGPUMemory();
        void show();
    public:
        static Eigen::Matrix3d cam;
        /// 参考帧的sift特征点
        SiftData lastSiftData;
        bool firstImage;
        bool firstOptimization;
    private:
        /// 移植的cudaSift函数
        int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
        /// 获取当前特征点的深度
        bool getDepth(SiftData& siftdata,cv::Mat depth);
        /// RANSAC
        pose RANSAC(SiftData& sift);
    private:
        double pointDepth[2048];
        /// 显示
        shared_ptr<visualization> vizShow;
        frame::ptr lastFrame;
        deque<frame::ptr> frames;
    };
}
#endif //FEATURETRACKING_FEATURETRACKING_H
