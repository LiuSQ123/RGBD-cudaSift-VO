//
// Created by liushiqi on 19-7-20.
//
#include "costFunction.h"
#include <iostream>
#include <cmath>
using namespace BA;
Eigen::Matrix3d BA::costFunction_reprojection::cam;

void costFunction_reprojection::getResidual(Eigen::Matrix<double ,2,1>& residualBlock){
    /// se(3)->旋转矩阵
    Eigen::Matrix4d Pose = Sophus::SE3::exp(*pPos).matrix();
    /// 特征点从世界坐标系转换到相机坐标系
    // Block of size (p,q), starting at (i,j)
    // matrix.block(i,j,p,q)->针对动态矩阵,matrix.block<p,q>(i,j)->针对静态矩阵
    Eigen::Vector3d Point_c = Pose.block<3,3>(0,0)*(*pPoint);
    Point_c = Point_c + Pose.block<3,1>(0,3);
    /// 重投影
    Eigen::Vector3d reprojection = costFunction_reprojection::cam*(Point_c);
    reprojection = reprojection/reprojection.z();
    /// 计算参差
    residualBlock = measurement - reprojection.topRows(2);
    /// 核函数
    double w = 1.0;
    double res_x = residualBlock(0,0);
    double res_y = residualBlock(1,0);
    double res_2 = res_x*res_x + res_y*res_y;
    double res = sqrt(res_2);
    if(res > 1.7){
        double kernal = log(1.0+res_2);
        //double kernal = res_2 /(1+res_2);
        kernal = sqrt(kernal);
        //w = kernal / res;
       // w = sqrt(w);
        residualBlock = w*residualBlock;
        //residualBlock = w*residualBlock;
        //cout<<w<<endl;
    }
}

bool costFunction_reprojection::Evaluate(
        Eigen::Matrix<double ,2,1>& residualBlock,
        Eigen::Matrix<double ,2,6>& posJacobian,
        Eigen::Matrix<double ,2,3>& pointJacobian){

    /// 判断非数 non inf
    /// se(3)->旋转矩阵
    Eigen::Matrix4d Pose = Sophus::SE3::exp(*pPos).matrix();
    /// 特征点从世界坐标系转换到相机坐标系
    // Block of size (p,q), starting at (i,j)
    // matrix.block(i,j,p,q)->针对动态矩阵,matrix.block<p,q>(i,j)->针对静态矩阵
    Eigen::Vector3d Point_c = Pose.block<3,3>(0,0)*(*pPoint);
    Point_c = Point_c + Pose.block<3,1>(0,3);
    //Eigen::Vector3d Pw = Pose.block<3, 3>(0, 0) * (Pc - Pose.block<3, 1>(0, 3));
    /// 重投影
    Eigen::Vector3d reprojection = costFunction_reprojection::cam*(Point_c);
    reprojection = reprojection/reprojection.z();
    /// 计算参差
    residualBlock = measurement - reprojection.topRows(2);
    //cout<<residualBlock.transpose()<<endl;

    /// 核函数
    double w = 1.0;
    double res_x = residualBlock(0,0);
    double res_y = residualBlock(1,0);
    double res_2 = res_x*res_x + res_y*res_y;
    double res = sqrt(res_2);
    if(res > 1.7){
        double kernal = log(1.0+res_2);
        //double kernal = res_2 /(1+res_2);
        kernal = sqrt(kernal);
        //w = kernal / res;
        //w = sqrt(w);
        residualBlock = w*residualBlock;
        //residualBlock = w*residualBlock;
        //cout<<w<<endl;
    }

    /// 计算关于姿态的jacobian
    double fx = cam(0,0);
    double fy = cam(1,1);
    double x = Point_c.x();
    double y = Point_c.y();
    double z = Point_c.z();

    posJacobian(0,0) = fx/z;
    posJacobian(0,1) = 0.0;
    posJacobian(0,2) =-1*((fx*x)/(z*z));
    posJacobian(0,3) =-1*((fx*x*y)/(z*z));
    posJacobian(0,4) = fx+(fx*x*x)/(z*z);
    posJacobian(0,5) =-1.0*((fx*y)/z);
    posJacobian(1,0) = 0;
    posJacobian(1,1) = fy/z;
    posJacobian(1,2) =-1.0*((fy*y)/(z*z));
    posJacobian(1,3) =-fy-((fy*y*y)/(z*z));
    posJacobian(1,4) = (fy*x*y)/(z*z);
    posJacobian(1,5) = (fy*x)/z;
    ///注意此处的-1,参见《14讲》p164
    posJacobian = -1.0*w*posJacobian;
    //posJacobian = -1*posJacobian;

    ///计算关于 特征点位置的jacobian
    pointJacobian(0,0) = fx/z;
    pointJacobian(0,1) = 0.0;
    pointJacobian(0,2) =-1.0*((fx*x)/(z*z));
    pointJacobian(1,0) = 0.0;
    pointJacobian(1,1) = fy/z;
    pointJacobian(1,2) =-1.0*((fy*y)/(z*z));
    ///注意此处的-1
    pointJacobian = -1.0*w*pointJacobian*Pose.block(0,0,3,3);
    //pointJacobian = -1*pointJacobian;
    return true;
}