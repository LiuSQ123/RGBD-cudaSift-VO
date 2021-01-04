//
// Created by liushiqi on 19-8-22.
//

#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <sstream>

#include <vector>

#include <thread>
#include <mutex>

#include "featureTracking.h"

using namespace std;
VO::featureTracking my_VO;
int main() {

    VO::featureTracking::cam = Eigen::Matrix3d::Zero();
    VO::featureTracking::cam(0,0)=525.0;
    VO::featureTracking::cam(1,1)=525.0;
    VO::featureTracking::cam(0,2)=319.5;
    VO::featureTracking::cam(1,2)=239.5;
    VO::featureTracking::cam(2,2)=1.0;
    BA::costFunction_reprojection::cam = VO::featureTracking::cam;

    //打开文件 读取数据
    ifstream imageFile;
    string dataDir = "/home/liushiqi/offline_SLAM_data/rgbd/rgbd_dataset_freiburg1_xyz";
    //string dataDir = "/home/liushiqi/offline_SLAM_data/rgbd/rgbd_dataset_freiburg2_desk";
    //string dataDir = "/home/liushiqi/offline_SLAM_data/rgbd/rgbd_dataset_freiburg1_desk";
    imageFile.open(dataDir + "/fr1_room.txt", ios::in);
    //imageFile.open(dataDir + "/desk.txt", ios::in);
    assert(imageFile.is_open());
    stringstream ss;
    string str;
    vector<vector<string>> images;
    while(getline(imageFile,str))
    {
        ss.clear();
        //cout<<str<<endl;
        int begin = 0;
        string time,rgb,depth;
        vector<string> strs;
        for(int i=0;i<str.size();++i){
            if(str[i] == ' ' || i == str.size()-1){
                if(i == str.size()-1)
                    strs.push_back(str.substr(begin,i-begin+1));
                else strs.push_back(str.substr(begin,i-begin));
                begin = i+1;
            }
        }
        images.push_back(strs);
    }
    imageFile.close();
    string lastImage ;
    for(auto& i:images){
        if(i.size()!=4) continue;
        cv::Mat rgb = cv::imread(dataDir+'/'+i[1],cv::IMREAD_GRAYSCALE);
        lastImage =dataDir+'/'+i[1];
        cv::Mat depth = cv::imread(dataDir+'/'+i[3]);
        //cv::imshow("a",depth);
        //cv::waitKey(0);
        ss.clear();
        ss<<i[0];
        double time ;
        ss>>time;
        cout<<"time: "<<time<<endl;
        my_VO.RGBDTracking(rgb,depth,time);
    }

    cv::Mat rgb = cv::imread(lastImage,cv::IMREAD_GRAYSCALE);
    cv::imshow("lastImage",rgb);
    cv::waitKey(0);
    return 0;
}