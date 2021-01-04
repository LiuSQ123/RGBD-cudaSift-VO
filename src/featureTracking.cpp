//
// Created by liushiqi on 19-7-31.
//
//#include "VO.h"
#include "featureTracking.h"
#include "cudaSift.h"

using namespace VO;
using namespace std;

Eigen::Matrix3d featureTracking::cam;

featureTracking::featureTracking(int _databaseSize){
    firstOptimization = true;
    firstImage = true;
    clearAllData();
    vizShow = make_shared<visualization>("test", 1);
}

void featureTracking::clearAllData(){
    //ID2ID.clear();
    /// vector 清零
    // currentFeatures3d.clear();
    // currentFeatures2d.clear();
}

void featureTracking::resetAllData(){
    firstImage = true;
}

bool featureTracking::RGBDTracking(cv::Mat image,cv::Mat depth,double time){
    /// 提取特征
    extractFeature(image,depth);
    optimization();
    show();
    /// 写入

    std::ofstream foutC("/home/liushiqi/offline_SLAM_data/rgbd/rgbd_dataset_freiburg1_xyz/mypath.csv", ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    /// 保留小数点后四位
    foutC.precision(4);
    foutC << time << " ";
    foutC.precision(5);
    Eigen::Matrix4d Pose = Sophus::SE3::exp(*(frames.front()->pos)).matrix();
    //Pose.block<3, 3>(0, 0).inverse() * (Pc - Pose.block<3, 1>(0, 3));
    Eigen::Quaterniond Q(Pose.block<3, 3>(0, 0));
    Eigen::Vector3d t = Pose.block<3, 1>(0, 3);

    foutC << t.x() << " "
          << t.y() << " "
          << t.z() << " "
          << Q.x() << " "
          << Q.y() << " "
          << Q.z() << " "
          << Q.w() << endl;
    foutC.close();

    slidWindow();
    return true;
}
featureTracking::pose featureTracking::RANSAC(SiftData& sift){
    /// 返回值是估算出来的位姿初始值

}
void featureTracking::show(){
    Eigen::Matrix4d lastPose = Sophus::SE3::exp(*(frames.front()->pos)).matrix();
    //matrix.block<p,q>(i,j)->针对静态矩阵
    vizShow->pushPOS(lastPose.block<3,3>(0,0),lastPose.block<3,1>(0,3));
}
bool featureTracking::extractFeature(cv::Mat image,cv::Mat depth){
    /// 传入的必须是灰度图
    if(image.channels()!= 1)
        return false;
    cv::Mat currentMat;
    image.convertTo(currentMat, CV_32FC1,1);
    /// 获取图像尺寸
    unsigned int w = currentMat.cols;
    unsigned int h = currentMat.rows;
    auto start = cv::getTickCount();
    /// 当前的sift特征点
    SiftData currentSiftData;
    /// 设定了2048个特征点的空间
    //InitSiftData(currentSiftData, 2048, true, true);
    InitSiftData(currentSiftData, 2048, true, true);

    /// 开辟显存空间并且上传到显存
    ///
    CudaImage currentImg;
    currentImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)currentMat.data);
    currentImg.Download();
    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    ExtractSift(currentSiftData, currentImg, 5, 1.0, 3.0, 0.0f, false, memoryTmp);

    FreeSiftTempMemory(memoryTmp);

    /// 当前帧与参考帧进行匹配
    /// 需不需要把滑动窗口内的所有帧进行匹配？？ for(int i=0;i<windowSize;++i){....}
    if(!firstImage)
        MatchSiftData(currentSiftData, lastSiftData);
        //MatchSiftData(lastSift, currentSiftData);
    /// ----------------------------------------------------------------------

    double time = (cv::getTickCount() - start) / cv::getTickFrequency();
    //std::cout<<"Finish ...... t =:"<<time*1000<<"ms"<<std::endl;
    /// 提取深度
    if(!depth.empty()) getDepth(currentSiftData,depth);
    /// 当前特征点的位姿指针
    posePtr currentPose = make_shared<featureTracking::pose>();
    /// 对其赋初值
    /// currentPose =
    /// 当前关键帧
    frame::ptr currentFrame = make_shared<frame>(currentPose);
    if(!firstImage){
        /// RANSAC,外点剔除
        /// 如果是第一帧图片,注意姿态的赋值
        float homography[9];
        int numMatches;
        FindHomography(currentSiftData, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
        ImproveHomography(currentSiftData, homography, 5, 0.00f, 0.80f, 3.0);

        *currentPose = *(lastFrame->pos);

    }
    else{
        /// 第一帧图片，姿态赋值为单位阵
        currentPose->setZero();
    }
    /// 注意内存泄露
    /// 设定当前大小
    //cout<<"firstImage:"<<firstImage<<endl;
    currentFrame->features.resize(currentSiftData.numPts);
    int eee = 0;
    int kkk = 0;
    int newPointCount =0;
    if(!firstImage){
        /// 清空匹配的对应关系
        for(auto& i:lastFrame->features)
            if(!i) continue;
            else i->IDinNextrame = -1;
    }
    for(int i=0;i<currentSiftData.numPts;++i){
        //if(!state[i]) continue;

        auto p = &currentSiftData.h_data[i];
        //bool flag = p->match_error > 3.0;
        bool flag = p->match_error > 2.0;
        /// 如果可以正确匹配，那么插入当前特征点库
        flag = flag | firstImage;
        if (!flag) {
            /// 如果匹配的指针是空的,那么建立新特征点
            /// 深度为0可能会产生空指针
            //cout<<lastFrame->features[p->match].use_count()<<endl;
            if (!lastFrame->features[p->match]){

                //cout<<"空"<<endl;
                if (pointDepth[i] < 0.0) continue;
                /// 首先计算出此特征点的世界坐标系坐标
                /// 如果没有深度，那么我们对其进行三角化(第一版暂不考虑这么做)
                Eigen::Matrix4d Pose = Sophus::SE3::exp(*currentPose).matrix();
                /// 计算当前相机坐标系下3d点的位置
                double z = pointDepth[i];
                double x = (p->xpos - cam(0, 2)) * z / cam(0, 0);
                double y = (p->ypos - cam(1, 2)) * z / cam(1, 1);
                //cout<<"x:"<<x<<", y:"<<y<<", z:"<<z<<endl;
                /// 世界坐标系下点的位置
                Eigen::Vector3d Pc(x, y, z);
                Eigen::Vector3d Pw = Pose.block<3, 3>(0, 0).inverse() * (Pc - Pose.block<3, 1>(0, 3));
                //Pw = 1.1*Pw;
                //cout<<Pw.transpose()<<endl;
                /// 计算结束
                shared_ptr<Eigen::Vector3d> currentPoint = make_shared<Eigen::Vector3d>(Pw);
                currentFrame->features[i] = make_shared<feature>(currentPoint, true);
                currentFrame->features[i]->measurement = Eigen::Vector2d(p->xpos, p->ypos);

                continue;
            }
            //cout << "不空" << endl;
            currentFrame->features[i] = make_shared<feature>(lastFrame->features[p->match]->point);
            currentFrame->features[i]->measurement = Eigen::Vector2d(p->xpos, p->ypos);
            /// 当前帧特与上一个关键帧特征的对应关系
            //currentFrame->features[i]->IDinLastFrame = p->match;
            /// 上一个特征帧与当前帧特征的对应关系
            /// 首先检测是否为空
            if(lastFrame->features[p->match])
                lastFrame->features[p->match]->IDinNextrame = i;
            cv::Point2d measurement(p->xpos, p->ypos);
            cv::circle(image, measurement, 5, cv::Scalar(0, 0, 255), 2);
        }
            /// 匹配错了就认为是一个新特征点
        else {
            //++newPointCount;
            /// 如果没有观测到深度，那么跳过(暂行)
            if (pointDepth[i] < 0.0) continue;
            /// 首先计算出此特征点的世界坐标系坐标
            /// 如果没有深度，那么我们对其进行三角化(第一版暂不考虑这么做)
            //Eigen::Matrix4d Pose = Sophus::SE3::exp(*currentPose).matrix();

            eee++;
            if(eee == 2){
                eee = 0;
                continue;
            }

            kkk++;
            if(kkk == 2){
                kkk = 0;
                continue;
            }
            ++newPointCount;
            Eigen::Matrix4d Pose = Sophus::SE3::exp(*currentPose).matrix();
            /// 计算当前相机坐标系下3d点的位置
            double z = pointDepth[i];
            double x = (p->xpos - cam(0, 2)) * z / cam(0, 0);
            double y = (p->ypos - cam(1, 2)) * z / cam(1, 1);
            //cout<<"x:"<<x<<", y:"<<y<<", z:"<<z<<endl;
            /// 世界坐标系下点的位置
            Eigen::Vector3d Pc(x, y, z);
            Eigen::Vector3d Pw = Pose.block<3, 3>(0, 0).inverse() * (Pc - Pose.block<3, 1>(0, 3));
            //Pw = 1.1*Pw;
            //cout<<Pw.transpose()<<endl;
            /// 计算结束
            shared_ptr<Eigen::Vector3d> currentPoint = make_shared<Eigen::Vector3d>(Pw);
            currentFrame->features[i] = make_shared<feature>(currentPoint, true);
            currentFrame->features[i]->measurement = Eigen::Vector2d(p->xpos, p->ypos);
            //cv::Point2d measurement(p->xpos, p->ypos);
            //cv::circle(image, measurement, 5, cv::Scalar(0, 0, 255), 2);
        }

    }
    //cv::imshow("image",image);
    //cv::waitKey(5);
    /// 判定关键帧
    if(newPointCount >= 100) currentFrame->isKeyFrame = true;
    if(frames.size()<5) currentFrame->isKeyFrame = true;

    currentFrame->pos = currentPose;
    /// 把当前帧插入队列中，如果不是关键帧，那么将会在滑动窗口中划出
    frames.push_back(currentFrame);
    /// 如果当前帧是关键帧，那么
    if(currentFrame->isKeyFrame = true) {
        lastFrame = currentFrame;
        /// free 不应在这里，放到滑动窗口中
        if (!firstImage)
            FreeSiftData(lastSiftData);
        lastSiftData = currentSiftData;
    }
    if(firstImage) firstImage = false;
    std::cout<<"last tracking point nums:"<<currentSiftData.numPts<<std::endl;

    //std::cout<<count<<std::endl;
    cv::imshow("Current Image",image);
    cv::waitKey(5);
    return true;
}

void featureTracking::slidWindow(){
    if(frames.size()<5) return;
    else{
        if(frames.back()->isKeyFrame){
            auto& oldFrame = frames.front();
            auto& secondOldFrame = frames[1];
            for(int j = 0;j<oldFrame->features.size();++j) {
                auto& feature = oldFrame->features[j];
                if(!feature) continue;
                int newID = feature->IDinNextrame;
                /// 如果没有匹配,跳过
                if(newID<0) continue;
                if(feature->isFirstObservation && feature->point.use_count()>1) {
                    /// 如果为空，那么跳过
                    if(!secondOldFrame->features[newID])
                        continue;
                    else {
                        secondOldFrame->features[newID]->isFirstObservation = true;
                    }
                }
            }
            frames.pop_front();
        }
        else frames.pop_back();
        //frames.pop_front();
    }
}

bool featureTracking::getDepth(SiftData& siftdata,cv::Mat depth){
    for(int i=0;i<siftdata.numPts;++i){
        cv::Point2d p(siftdata.h_data[i].xpos,siftdata.h_data[i].ypos);
        bool flag = ( depth.at<ushort>(p)/5000.0 > 0.0 && !isnan( depth.at<ushort>(p)) && !isinf( depth.at<ushort>(p)) );
        if(flag){
            pointDepth[i] = depth.at<ushort>(p)/5000.0;
            //cout<<"depth: "<<pointDepth[i]<<endl;
        }
        else pointDepth[i] = -1.0;
    }
    return true;
}

bool featureTracking::optimization(){

    BA::Problem myBA(15);
    if(frames.size() < 2 )
        return false;
    //cout<<"BA"<<endl;
    /// 添加姿态
    for(int i=0;i<frames.size();++i){
        myBA.addParameterBlock(frames[i]->pos);
        if(i==0) {
            //firstOptimization = false;
            myBA.setFixed(frames[i]->pos);
        }
    }
    /// 添加特征点位置
    int pointNum=0;
    for(int i=0;i<frames.size();++i){
        for(int j =0;j<frames[i]->features.size();++j) {
            auto& feature = frames[i]->features[j];
            /// 如果为空
            if(!feature) continue;
            /// 观测次数大于1才可以
            if(feature->isFirstObservation && feature->point.use_count()>2) {
                ++pointNum;
                myBA.addParameterBlock(feature->point);
                //if(i == 0)
                    //myBA.setFixed(feature->point);
            }
        }
    }
    cout<<"pointNum = :"<<pointNum<<endl;
    /// 针对Pos添加costfunction
    for(int i=0;i<frames.size();++i){
        /// 对每一帧上的所有特征点进行遍历
        for(int j =0;j<frames[i]->features.size();++j){
            auto& feature = frames[i]->features[j];
            /// 如果为空
            if(!feature) continue;
            if(feature->point.use_count()>2) {
                auto costFunction = make_shared<BA::costFunction_reprojection>(feature->measurement, feature->point,
                                                                               frames[i]->pos);
                myBA.addResidualBlock(costFunction);
            }
        }
    }
    //myBA.solveByGN();
    myBA.solveByLM();
    //for(int i=0;i<frames.size();++i){
        //Eigen::Matrix4d Pos_Matrix = Sophus::SE3::exp(*(frames[i]->pos)).matrix();
        //cout<<"POSE :"<<i<<endl;
        //cout<<Pos_Matrix<<endl;
   // }
    /*
    Eigen::Matrix4d Pos_Matrix = Sophus::SE3::exp(Pos1).matrix();
    cout<<Pos_Matrix<<endl;
    Pos_Matrix = Sophus::SE3::exp(Pos0).matrix();
    cout<<Pos_Matrix<<endl;
    */
}

int featureTracking::ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
#ifdef MANAGEDMEM
    SiftPoint *mpts = data.m_data;
#else
    if (data.h_data==NULL)
        return 0;
    SiftPoint *mpts = data.h_data;
#endif
    float limit = thresh*thresh;
    int numPts = data.numPts;
    cv::Mat M(8, 8, CV_64FC1);
    cv::Mat A(8, 1, CV_64FC1), X(8, 1, CV_64FC1);
    double Y[8];
    for (int i=0;i<8;i++)
        A.at<double>(i, 0) = homography[i] / homography[8];
    for (int loop=0;loop<numLoops;loop++) {
        M = cv::Scalar(0.0);
        X = cv::Scalar(0.0);
        for (int i=0;i<numPts;i++) {
            SiftPoint &pt = mpts[i];
            if (pt.score<minScore || pt.ambiguity>maxAmbiguity)
                continue;
            float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0f;
            float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
            float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
            float err = dx*dx + dy*dy;
            float wei = (err<limit ? 1.0f : 0.0f); //limit / (err + limit);
            Y[0] = pt.xpos;
            Y[1] = pt.ypos;
            Y[2] = 1.0;
            Y[3] = Y[4] = Y[5] = 0.0;
            Y[6] = - pt.xpos * pt.match_xpos;
            Y[7] = - pt.ypos * pt.match_xpos;
            for (int c=0;c<8;c++)
                for (int r=0;r<8;r++)
                    M.at<double>(r,c) += (Y[c] * Y[r] * wei);
            X += (cv::Mat(8,1,CV_64FC1,Y) * pt.match_xpos * wei);
            Y[0] = Y[1] = Y[2] = 0.0;
            Y[3] = pt.xpos;
            Y[4] = pt.ypos;
            Y[5] = 1.0;
            Y[6] = - pt.xpos * pt.match_ypos;
            Y[7] = - pt.ypos * pt.match_ypos;
            for (int c=0;c<8;c++)
                for (int r=0;r<8;r++)
                    M.at<double>(r,c) += (Y[c] * Y[r] * wei);
            X += (cv::Mat(8,1,CV_64FC1,Y) * pt.match_ypos * wei);
        }
        cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
    }
    int numfit = 0;
    for (int i=0;i<numPts;i++) {
        SiftPoint &pt = mpts[i];
        float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0;
        float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
        float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
        float err = dx*dx + dy*dy;
        if (err<limit)
            numfit++;
        pt.match_error = sqrt(err);
    }
    for (int i=0;i<8;i++)
        homography[i] = A.at<double>(i);
    homography[8] = 1.0f;
    return numfit;
}