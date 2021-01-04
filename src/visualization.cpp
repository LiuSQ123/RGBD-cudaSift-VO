//
// Created by liushiqi on 19-7-3.
//

#include "visualization.h"
visualization::visualization(std::string _windowName,int _loop_time){
    isend = false;
    isclose = false;
    loop_time = _loop_time;
    if(loop_time<=0){
        loop_time=10;
        std::cout<<RED<<"ERROR INPUT:loop_time set 10 ms."<<RESET<<std::endl;
    }
    windowName = _windowName;
    p_virtualWorld.reset(new cv::viz::Viz3d(windowName));
    p_virtualWorld->setBackgroundColor(cv::viz::Color::black());
    ///显示坐标系
    p_virtualWorld->showWidget("Coordinate", cv::viz::WCoordinateSystem(0.1));
    cv::Vec2d cam_pos(1.0, 1.0);
    p_virtualWorld->showWidget("camera_now",cv::viz::WCameraPosition(cam_pos,0.1,cv::viz::Color::yellow()));
    std::thread showThread(std::bind(&visualization::show,this));
    showThread.detach();

}

bool visualization::close(){
    isclose = true;
    while(!isend) usleep(1000);
    return isend;
}

visualization::~visualization(){
    close();
    p_virtualWorld->close();
}

void visualization::pushPOS(const Eigen::Matrix3d& R,const Eigen::Vector3d& t){
    ///显示当前位姿
    mutex.lock();
    cv::Vec3f cv_t(t(0),t(1),t(2));
    cv::Mat cv_mat;
    cv::eigen2cv(R,cv_mat);
    cv::Affine3d pose(cv_mat, cv_t);
    p_virtualWorld->setWidgetPose("camera_now",pose);
    mutex.unlock();
    ///结束
    cam_list.emplace_back(cv::Vec2d(1.0, 1.0),0.5,cv::viz::Color::green());
    //showCam(cam_list.size()-1,R,t);
    R_list.emplace_back(R);
    //添加一条线
    if(t_list.empty()) line_list.emplace_back(cv::Vec3d(0,0,0), cv::Vec3d(t(0),t(1),t(2)), cv::viz::Color::green());
    else line_list.emplace_back(cv::Vec3d(t_list.back()(0),t_list.back()(1),t_list.back()(2)), cv::Vec3d(t(0),t(1),t(2)), cv::viz::Color::green());
    ///显示轨迹
    showLineBetweenCam(line_list.size()-1);
    t_list.emplace_back(t);
    ///刷新显示
    refreshWindow();
}

void visualization::pushPointCloud(const Eigen::Vector3d& P){
    Point_list.emplace_back(P);
    ///刷新界面
    refreshWindow();
}

void visualization::cleanPOS(){
    R_list.clear();
    ///刷新界面
    refreshWindow();
}

void visualization::cleanPoint(){
    Point_list.clear();
    ///刷新界面
    refreshWindow();
}

void visualization::show(){
    std::cout<<GREEN<<"DISPLAY thread start......"<<RESET<<std::endl;
    while(!isclose){
        mutex.lock();
        p_virtualWorld->spinOnce(loop_time, true);
        mutex.unlock();
        usleep(1*100);
    }
    mutex.lock();
    //p_virtualWorld->removeAllWidgets();
    mutex.unlock();
    isend = true;
    std::cout<<GREEN<<"DISPLAY thread end......"<<RESET<<std::endl;
}

void visualization::showCam(const int id,const Eigen::Matrix3d& _R,const Eigen::Vector3d& _t){
    //添加一个相机
    cv::Vec3d t(_t.x(),_t.y(),_t.z());
    cv::Mat r;
    cv::eigen2cv(_R,r);
    cv::Affine3d pose(r, t);

    mutex.lock();
    p_virtualWorld->showWidget("camera"+std::to_string(id),cam_list[id]);
    p_virtualWorld->setWidgetPose("camera"+std::to_string(id),pose);
    mutex.unlock();
}

void visualization::showLineBetweenCam(const int id){
    mutex.lock();
    p_virtualWorld->showWidget("line"+std::to_string(id),line_list[id]);
    mutex.unlock();
}

void visualization::showKeyFrame(){
    if(R_list.size()!=cam_list.size()){
        std::cout<<RED<<"fatal erroe:R_list.size()!=cam_list.size()"<<RESET<<std::endl;
        close();
        return;
    }
    for (int i = 0; i < R_list.size(); ++i)
        showCam(i, R_list[i], t_list[i]);
}

void visualization::removeAllKeyFrame(){
    if(R_list.size()!=cam_list.size()){
        std::cout<<RED<<"fatal erroe:R_list.size()!=cam_list.size()"<<RESET<<std::endl;
        close();
        return;
    }
    mutex.lock();
    for(int i=0;i<R_list.size();++i)
        p_virtualWorld->removeWidget("camera"+std::to_string(i));
    mutex.unlock();
}

void visualization::refreshAllKeyFrame(){
    removeAllKeyFrame();
    showKeyFrame();
}

void visualization::removeAllLine(){

    if(line_list.size()!=t_list.size()){
        std::cout<<RED<<"fatal erroe:line_list.size()!=t_list.size()"<<RESET<<std::endl;
        close();
        return;
    }
    mutex.lock();
    for(int i=0;i<line_list.size();++i)
        p_virtualWorld->removeWidget("line"+std::to_string(i));
    mutex.unlock();
}

void visualization::showAllLine(){
    if(line_list.size()!=t_list.size()){
        std::cout<<RED<<"fatal erroe:line_list.size()!=t_list.size()"<<RESET<<std::endl;
        close();
        return;
    }
    for(int i=0;i<line_list.size();++i)
        showLineBetweenCam(i);
}

void visualization::refreshAllLine(){
    removeAllLine();
    showAllLine();
}

void visualization::refreshWindow(){
    //p_virtualWorld->spinOnce(loop_time, true);
}