//
// Created by liushiqi on 19-8-11.
//

#ifndef TESTCUDA_FRAME_H
#define TESTCUDA_FRAME_H
#include <eigen3/Eigen/Core>
#include <vector>
#include <memory>
#include <deque>
#include <unordered_map>
#include <unordered_set>

//#include "featureTracking.h"
#include "feature.h"
using namespace std;
/// 帧
class frame{
public:
    typedef std::shared_ptr<frame> ptr;
    frame(shared_ptr<Eigen::Matrix<double,6,1>>& _pos):pos(_pos),isKeyFrame(false){};
    ~frame() = default;
    bool isKeyFrame;
    /// 该关键帧上的所有特征点的指针
    std::vector<feature::ptr> features;
    //std::unordered_map<int,feature::ptr> features;
    /// 该帧的位置
    shared_ptr<Eigen::Matrix<double,6,1>> pos;
};
/// 关键帧
class keyFrame{
public:
    /// 关键帧指针
    typedef std::shared_ptr<keyFrame> ptr;
    keyFrame(Eigen::Matrix3d& R,Eigen::Vector3d& t);
    ~keyFrame() = default;
    bool isKeyFrame;
    ///该关键帧上的所有特征点的指针
    std::vector<feature::ptr> points;
    ///该帧的位置
    Eigen::Matrix<double,6,1>* pos;
};

#endif //TESTCUDA_FRAME_H
