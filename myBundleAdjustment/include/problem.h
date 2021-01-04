//
// Created by liushiqi on 19-7-19.
//

#ifndef MYBUNDLEADJUSTMENT_PROBLEM_H
#define MYBUNDLEADJUSTMENT_PROBLEM_H

#include "costFunction.h"
#include "color.h"
#include "lossFunction.hpp"
#include <memory>
#include <deque>
#include <vector>
#include <unordered_map>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseQR>
#include <eigen3/Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>

#include <sophus/se3.h>

#define FIXED true
#define FLEXIBLE false

namespace BA{
    using namespace std;
    class Problem{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<Eigen::Vector3d> ptr3D;
        typedef shared_ptr<Eigen::Matrix<double ,6,1>> ptrPose;
        typedef shared_ptr<costFunction_reprojection> ptrCost;

        typedef Eigen::SparseMatrix<double> SM;
        typedef Eigen::Triplet<double> TRI;
        ///最大迭代次数默认是5
        Problem(int _max_loop = 8);

        ///添加状态量
        void addParameterBlock(ptr3D Point); //添加特征点
        void addParameterBlock(ptrPose Pose); //添加姿态
        ///添加观测量
        void addResidualBlock(ptrCost costFunction,lossFunction* lossFunction = NULL); //添加代价函数
        void setFixed(ptrPose Pose);
        void setFixed(ptr3D Point);
        ///高斯牛顿法求解
        bool solveByGN();
        ///LM法求解
        bool solveByLM();
        /// 使用图片展现矩阵
        void showMatrixAsImage(Eigen::MatrixXd& input,string windowName="Matrix");
        void showMatrixAsImage(SM& input,string windowName="Matrix");
        void showBriefReport();
    private:
        Eigen::MatrixXd solveHByFactorization(SM& H,Eigen::VectorXd& g);
        Eigen::MatrixXd solveHBySolveBlock(SM& H,Eigen::VectorXd& g);
        //void sortEverCostfunction();
        //void pointSort(vector<ptrCost>& Pose_costFunction,int _low,int _high);
        void findJacobian(ptrPose Pose, vector<ptrCost>& Pose_costFunction,Eigen::MatrixXd& jacobian,Eigen::MatrixXd& residual);
        /// 求解稀疏雅克比矩阵
        void findSparseJacobian(ptrPose& Pose, vector<ptrCost>& Pose_costFunction,vector<TRI>& jacobian,vector<double >& residual,int& rowNow);
        /// 回滚为原先的状态
        void rollBack();
        void upDate(ptrPose x,Eigen::Matrix<double ,6,1> delt_x);
        void upDate(ptr3D x,Eigen::Vector3d delt_x);
    public:
        int MAX_LOOP;
    private:
        /// 迭代过程中的参数
        /// μ,v,τ
        double miu;
        double v;
        double tao;
        /// 记录回滚状态
        Eigen::MatrixXd lastX;
        /// 记录坐标
        int countPoint;
        int countPose;
        int fixedPointNow; // 记录当前共有多少个特征点固定
        int fixedPoseNow;  // 记录当前共有多少个姿态被固定
        int flexiblePoseNum;  ///需要优化的特征点和姿态数值
        int flexiblePointNum;
        int residualNUM;
        ///姿态和cost_function的对应关系
        unordered_map<ptrPose,vector<ptrCost>>  Pose_costFunction;
        /// ID->特征点，位姿的映射
        unordered_map<int,ptr3D> ID2point;
        unordered_map<int,ptrPose> ID2pose;
        /// 记录旧坐标
        unordered_map<int,Eigen::Vector3d> ID2oldPoint;
        unordered_map<int,Eigen::Matrix<double ,6,1>> ID2oldPose;
        /// 特征点->ID 的映射
        unordered_map<ptr3D, int> point2ID;
        unordered_map<ptrPose, int> pose2ID;
        /// 状态量固定标志位
        unordered_map<ptrPose ,bool> fixPoseFlag;
        unordered_map<ptr3D,bool > fixPointFlag;

    };
}

#endif //MYBUNDLEADJUSTMENT_PROBLEM_H
