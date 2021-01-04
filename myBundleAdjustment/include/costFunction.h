//
// Created by liushiqi on 19-7-19.
//

#ifndef MYBUNDLEADJUSTMENT_COSTFUNCTION_H
#define MYBUNDLEADJUSTMENT_COSTFUNCTION_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <sophus/se3.h>
#include <memory>
using namespace std;
namespace BA{
    ///重投影误差
    class costFunction_reprojection{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef  shared_ptr<Eigen::Vector3d> ptr3D;
        typedef  shared_ptr<Eigen::Matrix<double ,6,1>> ptrPose;
        /// _pPoint 为世界坐标系下的点
        costFunction_reprojection(Eigen::Vector2d& _measurement,ptr3D& _pPoint,ptrPose& _pPos):
        measurement(_measurement),pPoint{_pPoint},pPos(_pPos){}
        bool Evaluate(Eigen::Matrix<double ,2,1>& residualBlock,Eigen::Matrix<double ,2,6>& posJacobian,Eigen::Matrix<double ,2,3>& pointJacobian);
        //bool Evaluate(Eigen::Matrix<double ,2,1>& residualBlock,vector<Eigen::Triplet<double>>& );
        void getResidual(Eigen::Matrix<double ,2,1>& residualBlock);
        ptr3D pPoint; // 特征点坐标
        ptrPose pPos;   // 姿态
        ///相机模型
        static Eigen::Matrix3d cam;
    private:
        Eigen::Vector2d  measurement;

    };
    ///预计分误差
    class costFunction_preintegration{};

}
#endif //MYBUNDLEADJUSTMENT_COSTFUNCTION_H
