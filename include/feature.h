//
// Created by liushiqi on 19-8-11.
//

#ifndef TESTCUDA_FEATURE_H
#define TESTCUDA_FEATURE_H

#include <eigen3/Eigen/Core>
#include <vector>
#include <memory>
#include <deque>
#include <unordered_map>
#include <unordered_set>

using namespace std;
/// 特征点
class feature{
public:
    /// feature指针
    typedef std::shared_ptr<feature> ptr;
    feature():isEmpty(true),IDinLastFrame(-1),IDinNextrame(-1){};
    ~feature() = default;
    //feature(int _ID,shared_ptr<Eigen::Vector3d> _point):ID(_ID),point(_point){}
    feature(shared_ptr<Eigen::Vector3d>& _point,bool flag = false):point(_point),isFirstObservation(flag),isEmpty(false){}
    bool isFirstObservation;
    bool isEmpty;
    /// 该特征点与上一帧关键帧匹配特征点的ID
    int IDinLastFrame;
    /// 该特征点与下一帧关键帧匹配特征点的ID
    int IDinNextrame;
    /// 该点的3d坐标的指针
    shared_ptr<Eigen::Vector3d> point;
    /// 特征点的一次观测
    Eigen::Vector2d measurement;
    /// 看到该特征点的所有关键帧
    // vector<std::shared_ptr<keyFrame>> keyFrames;
    /// 该特征点所有的观测
    // unordered_map<shared_ptr<Eigen::Matrix<double,6,1>>,Eigen::Vector2d> measurements;
};
#endif //TESTCUDA_FEATURE_H
