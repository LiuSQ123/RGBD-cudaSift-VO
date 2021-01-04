#include <iostream>
#include "problem.h"
#include "costFunction.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;
using namespace cv;
int main() {
    std::cout << "BA开始......" << std::endl;
    string imag1 = "/home/liushiqi/ClionProjects/BA/cmake-build-debug/11.png";
    string imag2 = "/home/liushiqi/ClionProjects/BA/cmake-build-debug/33.png";
    string imag3 = "/home/liushiqi/ClionProjects/BA/cmake-build-debug/111.png";
    //-- 设置相机内参
    Eigen::Matrix3d camMatrix;
    camMatrix.setZero();
    camMatrix(0,0)=525.0;
    camMatrix(1,1)=525.0;
    camMatrix(0,2)=319.5;
    camMatrix(1,2)=239.5;
    camMatrix(2,2)=1.0;
    BA::costFunction_reprojection::cam = camMatrix;
    //-- 读取图像
    Mat img_1 = imread (imag1);
    Mat img_2 = imread (imag2);
    Mat img_depth = imread(imag3);
    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create(500);
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第一步:检测 Oriented FAST 角点位置
    double t = (double)cv::getTickCount();
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );
    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    matcher->match ( descriptors_1, descriptors_2, matches );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    printf("ORB detect cost %f ms \n", (1000*(cv::getTickCount() - t) / cv::getTickFrequency()));
    cout<<"good_match = "<<good_matches.size()<<endl;

    std::vector<cv::Point2d> points1,points2;
    std::vector< DMatch > good_matches2;
    std::vector<cv::Point3d> v_3DPoints;
    std::vector<shared_ptr<Eigen::Vector3d>> E3DPoints;
    for ( int i = 0; i < ( int ) good_matches.size(); i++ )
    {
        ///
        if(img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)/5000!=0 &&
           !isnan( img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)) &&
           !isinf( img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)))
        {
            cv::Point3d temp;
            double  u=keypoints_1[good_matches[i].queryIdx].pt.x;
            double  v=keypoints_1[good_matches[i].queryIdx].pt.y;
            temp.z = img_depth.at<ushort>(keypoints_1[good_matches[i].queryIdx].pt)/5000.0;
            //if(temp.z > 2.0) continue;
            //if(temp.z==0) temp.z = 1;
            temp.x=(u-camMatrix(0,2))*temp.z/camMatrix(0,0);
            temp.y=(v-camMatrix(1,2))*temp.z/camMatrix(1,1);
            //cout<<temp<<endl;

            points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
            good_matches2.emplace_back(good_matches[i]);

            //求解3D点坐标，参见TUM数据集
            v_3DPoints.emplace_back(temp);
            //Eigen::Vector3d tt(temp.x,temp.y,temp.z);
            E3DPoints.push_back(make_shared<Eigen::Vector3d>(temp.x,temp.y,temp.z));
        }
    }
    cout<<"v_3DPoints.size:"<<v_3DPoints.size()<<endl;
    //使用OpenCV提供的代数方法求解：2D-2D
    Point2d principal_point ( 319.5, 239.5);	//相机光心, TUM dataset标定值
    double focal_length = 525;			        //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cv::Mat R,tt;
    recoverPose ( essential_matrix, points1, points2, R, tt, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<tt<<endl;
    // -- 开始BA优化求解：3D-2D
    BA::Problem myBA(12);
    Eigen::Matrix3d p;
    Eigen::Vector3d t2(-0.033,0.007,-0.126);
    cv::cv2eigen(R,p);

    ///设定Pos0
    //typedef  shared_ptr<Eigen::Vector3d> ptr3D;
    //typedef  shared_ptr<Eigen::Matrix<double ,6,1>> ptrPose;
    auto Pos0 = make_shared<Eigen::Matrix<double ,6,1>>();
    Pos0->setZero();
    ///添加姿态
    //myBA.addParameterBlock(Pos0);
    ///固定Pos0 坐标
    //myBA.setFixed(Pos0);

    ///设定Pos1;
    auto Pos1 = make_shared<Eigen::Matrix<double ,6,1>>();
    Pos1->setZero();
    ///设定初始值
    Sophus::SE3 new_Pos_se1 = Sophus::SE3(p, t2);
    *Pos1 = new_Pos_se1.log();
    Pos1->setZero();
    /// 添加姿态
    myBA.addParameterBlock(Pos1);
    //myBA.setFixed(Pos1);
    /// 添加特征点位置
    for(int i=0;i<E3DPoints.size();++i){
        myBA.addParameterBlock(E3DPoints[i]);
        myBA.setFixed(E3DPoints[i]);
    }

    /// 针对Pos0添加costfunction
    /*
    for(int i=0;i<points1.size();++i){
        Eigen::Vector2d m(points1[i].x,points1[i].y);
        auto temp = make_shared<BA::costFunction_reprojection>(m,E3DPoints[i],Pos0);
        myBA.addResidualBlock(temp);
    }
     */
    /// 针对Pos1添加costfunction
    for(int i=0;i<points2.size();++i){
        Eigen::Vector2d m(points2[i].x,points2[i].y);
        auto temp = make_shared<BA::costFunction_reprojection>(m,E3DPoints[i],Pos1);
        myBA.addResidualBlock(temp);
    }

    //myBA.solveByGN();
    myBA.solveByLM();
    Eigen::Matrix4d Pos_Matrix = Sophus::SE3::exp(*Pos1).matrix();
    cout<<Pos_Matrix<<endl;
    Pos_Matrix = Sophus::SE3::exp(*Pos0).matrix();
    cout<<Pos_Matrix<<endl;

    return 0;
}