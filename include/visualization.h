//
// Created by liushiqi on 19-7-3.
//

#ifndef VIZ_SHOW_VISUALIZATION_H
#define VIZ_SHOW_VISUALIZATION_H

#include <iostream>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>

#include <memory>
#include <unistd.h>
#include <list>
#include <vector>
#include <string>

#include <opencv2/viz/viz3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/viz/types.hpp>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include <opencv2/core/eigen.hpp>

#include "color.h"

class visualization {
public:
    /**
     * 构造函数
     * @param _hz 为显示界面刷新频率
     */
    visualization(std::string windowName,int _hz);
    ~visualization();
    void removeAllLine();
    void showAllLine();
    void refreshAllLine();
    /**
     * 显示所有关键帧
     */
    void showKeyFrame();
    /***
     * 隐藏显示所有关键帧
     */
    void removeAllKeyFrame();
    ///刷新关键帧坐标
    void refreshAllKeyFrame();
    /**
     * 插入位姿
     * @param R  旋转矩阵
     * @param t  位移
     */
    void pushPOS(const Eigen::Matrix3d& R,const Eigen::Vector3d& t);
    /**
     * 插入特征点
     * @param P 特征点坐标
     */
    void pushPointCloud(const Eigen::Vector3d& P);
    /**
     * 清除所有位姿
     */
    void cleanPOS();
    /**
     * 清除所有特征点
     */
    void cleanPoint();
    bool close();
    visualization()= delete; //禁止使用
public:

private:
    void refreshWindow();
    void show();
    void showCam(const int id,const Eigen::Matrix3d& _R,const Eigen::Vector3d& _t);
    void showLineBetweenCam(const int id);
private:
    std::shared_ptr<cv::viz::Viz3d> p_virtualWorld;
    bool isend;
    bool isclose;
    std::mutex mutex;
    int loop_time;
    std::string windowName;
    std::vector<Eigen::Vector3d> Point_list;   //特征点的坐标
    std::vector<Eigen::Matrix3d> R_list;       //旋转矩阵
    std::vector<Eigen::Vector3d> t_list;       //位移

    std::vector<cv::viz::WCameraPosition> cam_list; //相机
    std::vector<cv::viz::WLine> line_list;          //线段
    std::vector<cv::viz::WCloud> cloud_list;        //点云

};


#endif //VIZ_SHOW_VISUALIZATION_H
