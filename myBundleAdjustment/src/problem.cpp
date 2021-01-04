//
// Created by liushiqi on 19-7-20.
//
#include "problem.h"
#include <iostream>
#include <limits>

using namespace BA;
Problem::Problem(int _max_loop) {
    MAX_LOOP = _max_loop;
    countPoint = 0 ;
    countPose = 0;
    residualNUM = 0;
    miu = 1.0;
    v = 2.0;
    tao = 1e-6;
}
/// 添加参数的时候默认是不固定
void Problem::addParameterBlock(ptr3D Point){
    /// ID->特征点，位姿的映射
    ID2point[countPoint] = Point;
    /// 特征点->ID 的映射
    point2ID[Point] = countPoint;
    ++countPoint;
}

void Problem::addParameterBlock(ptrPose Pose){
    ID2pose[countPose] = Pose;
    pose2ID[Pose] = countPose;
    ++countPose;
}

void Problem::addResidualBlock(ptrCost  costFunction,lossFunction* lossFunction) {
    ///添加姿态与costFunction的映射关系
    Pose_costFunction[costFunction->pPos].push_back(costFunction);
    ++residualNUM;
}

void Problem::setFixed(ptrPose Pose){
    fixPoseFlag[Pose] = true;
#ifdef DEBUG
    std::cout<<YELLOW<<"添加固定姿态！"<<RESET<<std::endl;
#endif
}

void Problem::setFixed(ptr3D Point){
    fixPointFlag[Point] = true;
#ifdef DEBUG
    std::cout<<YELLOW<<"添加固定点！"<<RESET<<std::endl;
#endif
}

/*
/// 似乎根本不需要排序？？
/// 根据ID递增顺序对特征点快速排序
void Problem::pointSort(vector<ptrCost>& Pose_costFunction,int _low,int _high){
    if(_low>=_high) return;
    if(Pose_costFunction.empty()) return;
    int low = _low;
    int high = _high;
    ///key 的 id
    int key = point2ID[Pose_costFunction[low]->pPoint];
    auto t = Pose_costFunction[low];
    while(low<high){
        while(low<high && point2ID[Pose_costFunction[high]->pPoint] > key) --high;
        if(low<high){
            Pose_costFunction[low++] = Pose_costFunction[high];
        }
        while(low<high && point2ID[Pose_costFunction[low]->pPoint] < key) ++low;
        if(low<high){
            Pose_costFunction[high--] = Pose_costFunction[low];
        }
    }
    Pose_costFunction[low] = t;
    pointSort(Pose_costFunction,_low,low-1);
    pointSort(Pose_costFunction,low+1,_high);

}
/// 可能会出现一个特征点观测次数特别少的情况，这时候，我不准备优化这个特征点。
/// 先对所有观测的特征点做一次排序，按id由小到大排序
void Problem::sortEverCostfunction(){
    cout<<GREEN<<"特征点快速排序开始..."<<endl;
    for(auto& i:Pose_costFunction){
        pointSort(i.second,0,i.second.size()-1);
    }
    cout<<GREEN<<"特征点快速排序结束..."<<RESET<<endl;
}
*/

void Problem::findJacobian(ptrPose Pose,
        vector<ptrCost>& Pose_costFunction,
        Eigen::MatrixXd& jacobian,Eigen::MatrixXd& residual){

#ifdef DEBUG
    std::cout<<GREEN<<"Problem::findJacobian()"<< RESET<<std::endl;
#endif
    jacobian.resize(0,0);
    jacobian.setZero();
    residual.resize(0,0);
    residual.setZero();

    /// 如果这一帧特征点为空,那么报错
    if(Pose_costFunction.empty()) {
        cout<<RED<<"ERROE:Pose_costFunction.empty():Problem::findJacobian"<<RESET<<endl;
        return;
    }
    int numOfResidual=(int)Pose_costFunction.size();
    /// 克比矩阵的rows只和残差的大小有关，总是2*Residual
    jacobian.resize(2*numOfResidual,6*flexiblePoseNum+3*flexiblePointNum);
    jacobian.setZero();
    /// 残差
    residual.resize(2*numOfResidual,1);
    residual.setZero();
    /// 雅克比,残差 赋值
    int col_pose = 6*(pose2ID[Pose] - fixedPoseNow);
    int row_pose = 0;
    int row_res = 0;
    for(int i=0;i<Pose_costFunction.size();i++){
        //Block of size (p,q), starting at (i,j)
        //matrix.block(i,j,p,q)->针对动态矩阵,matrix.block<p,q>(i,j)->针对静态矩阵
        Eigen::Matrix<double ,2,1> residualBlock;
        Eigen::Matrix<double ,2,6> posJacobian;
        Eigen::Matrix<double ,2,3> pointJacobian;
        Pose_costFunction[i]->Evaluate(residualBlock,posJacobian,pointJacobian);
        auto pose = Pose_costFunction[i]->pPos;
        auto point = Pose_costFunction[i]->pPoint;
        ///如果姿态没有被约束，那么填入雅克比
        //cout<<"row:"<<posJacobian.rows()<<", col:"<<posJacobian.cols()<<endl;
        //cout<<"row_pose:"<<row_pose<<", col_pose:"<<col_pose<<endl;
        //cout<<"rows:"<<2*numOfResidual<<endl;
        if(!fixPoseFlag[pose])
            jacobian.block(row_pose,col_pose,2,6) = posJacobian;
        /// 如果特征点没有被约束，那么填入雅克比矩阵，否则该点的雅克比矩阵为0矩阵
        if(!fixPointFlag[point])
            jacobian.block(row_pose,6*flexiblePoseNum + 3*point2ID[point],2,3) = pointJacobian;
        /// 注意不可同时约束姿态和特征点，这样会造成H矩阵不可逆。
        /// 从另一方面说，如果同时约束姿态和特征点，那么也就没有必要优化这些姿态和特征点了，这些残差也就毫无意义。
        //ans.block(2*i,0,2,6)=findPoseJacobian(Pose,x.block(6+3*i,0,3,1));
        //ans.block(i*2,6+3*i,2,3)=findPointJacobian(Pose,x.block(6+3*i,0,3,1));
        residual.block(row_res,0,2,1) = residualBlock;
        row_pose += 2;
        row_res += 2;
    }
#ifdef DEBUG
    std::cout<<GREEN<<"Problem::findJacobian() end"<< RESET<<std::endl;
#endif
}

void Problem::findSparseJacobian(ptrPose &Pose,
        vector<ptrCost>& Pose_costFunction,
        vector<TRI>& jacobian,vector<double >& residual,int& rowNow){

#ifdef DEBUG
    std::cout<<GREEN<<"Problem::findSparseJacobian()"<< RESET<<std::endl;
#endif
    /// 如果这一帧特征点为空,那么报错
    if(Pose_costFunction.empty()) {
        cout<<RED<<"ERROE:Pose_costFunction.empty():Problem::findSparseJacobian"<<RESET<<endl;
        return;
    }
    /// 雅克比,残差 赋值
    /// 当前的行数和列数
    int colNow = 6*(pose2ID[Pose] - fixedPoseNow);

    for(int i=0;i<Pose_costFunction.size();i++){
        //Block of size (p,q), starting at (i,j)
        //matrix.block(i,j,p,q)->针对动态矩阵,matrix.block<p,q>(i,j)->针对静态矩阵
        Eigen::Matrix<double ,2,1> residualBlock;
        Eigen::Matrix<double ,2,6> posJacobian;
        Eigen::Matrix<double ,2,3> pointJacobian;
        Pose_costFunction[i]->Evaluate(residualBlock,posJacobian,pointJacobian);
        auto& pose = Pose_costFunction[i]->pPos;
        auto& point = Pose_costFunction[i]->pPoint;
        ///如果姿态没有被约束，那么填入雅克比
        /// 姿态的雅克比
        if(!fixPoseFlag[pose]) {
            jacobian.emplace_back(rowNow,colNow    ,posJacobian(0,0));
            jacobian.emplace_back(rowNow,colNow + 2,posJacobian(0,2));
            jacobian.emplace_back(rowNow,colNow + 3,posJacobian(0,3));
            jacobian.emplace_back(rowNow,colNow + 4,posJacobian(0,4));
            jacobian.emplace_back(rowNow,colNow + 5,posJacobian(0,5));

            jacobian.emplace_back(rowNow + 1,colNow + 1,posJacobian(1,1));
            jacobian.emplace_back(rowNow + 1,colNow + 2,posJacobian(1,2));
            jacobian.emplace_back(rowNow + 1,colNow + 3,posJacobian(1,3));
            jacobian.emplace_back(rowNow + 1,colNow + 4,posJacobian(1,4));
            jacobian.emplace_back(rowNow + 1,colNow + 5,posJacobian(1,5));
        }
        /// 如果特征点没有被约束，那么填入雅克比矩阵，否则该点的雅克比矩阵为0矩阵
        if(!fixPointFlag[point]) {
            int pointCol = 6 * flexiblePoseNum + 3 * point2ID[point];
            jacobian.emplace_back(rowNow ,pointCol    ,pointJacobian(0,0));
            jacobian.emplace_back(rowNow ,pointCol + 1,pointJacobian(0,1));
            jacobian.emplace_back(rowNow ,pointCol + 2,pointJacobian(0,2));

            jacobian.emplace_back(rowNow + 1,pointCol    ,pointJacobian(1,0));
            jacobian.emplace_back(rowNow + 1,pointCol + 1,pointJacobian(1,1));
            jacobian.emplace_back(rowNow + 1,pointCol + 2,pointJacobian(1,2));
        }
        /// 填入残差
        residual.push_back(residualBlock(0,0));
        residual.push_back(residualBlock(1,0));
        /// 更新行数
        rowNow += 2;
        /// 注意不可同时约束姿态和特征点，这样会造成H矩阵不可逆。
        /// 从另一方面说，如果同时约束姿态和特征点，那么也就没有必要优化这些姿态和特征点了，这些残差也就毫无意义。
    }
#ifdef DEBUG
    std::cout<<GREEN<<"Problem::findSparseJacobian() end"<< RESET<<std::endl;
#endif
}

bool Problem::solveByGN(){
    /// 首先对每一个特征点进行排序
    /// 不需要排序
    //sortEverCostfunction();

    /// 0.构造状态量 x
    /// 1.构造雅克比矩阵,求解f(x)
    /// 2.求解雅克比的转置矩阵
    /// 3.构造H矩阵
    /// 4.求解g
    /// 5.求解H的逆
    /// 6.求解δx
    /// 7.判断停止条件

    ///开始计时
    auto BAstart = cv::getTickCount();
    /// 保存Cost
    double currentCost = 0.;
    double lastCost = 0.;

#ifdef DEBUG
    std::cout<<"位姿数量："<< Pose_costFunction.size()<<std::endl;
    std::cout<<"特征数量："<< ID2point.size()<<std::endl;
#endif
#ifdef DEBUG
    std::cout<<YELLOW<<"-------------------------------0.构造状态量 x----------------------------------"<< RESET<<std::endl;
#endif
    Eigen::MatrixXd x;
    ///优化变量的大小
    flexiblePoseNum  = (int)ID2pose.size() - (int)fixPoseFlag.size();
    flexiblePointNum = (int)ID2point.size() - (int)fixPointFlag.size();

    if(flexiblePointNum==0) std::cout<<GREEN<<"只优化关键帧位置..."<<RESET<<std::endl;
    if(flexiblePoseNum==0) std::cout<<GREEN<<"只优化特征点位置..."<<RESET<<std::endl;
    if(flexiblePoseNum==0 && flexiblePointNum==0) {
        std::cout<<RED<<"错误：优化变量为0..."<<RESET<<std::endl;
        return false;
    }
#ifdef DEBUG
    std::cout << YELLOW << "------------"<<RED<<"开始迭代～～"<<BLUE<<"1.构造雅克比矩阵"<<GREEN<<"求解f(x)"<<YELLOW<<"-------------" << RESET << std::endl;
#endif
    for(int it =0;it<MAX_LOOP;++it) {
#ifdef DEBUG
        std::cout << YELLOW << "------------1.构造雅克比矩阵,求解f(x)-------------" << RESET << std::endl;
#endif
        /// 雅克比矩阵大小计算d  x
        Eigen::MatrixXd J;
        Eigen::MatrixXd fx;
        /// 雅克比矩阵的列数和优化变量的维度有关
        int J_cols = 6 * flexiblePoseNum + 3 * flexiblePointNum;
        if (J_cols == 0) {
            cout << RED << "ERROR:Problem::solveByGN..." << "J_cols=0" << endl;
            break;
        }
        /// 雅克比矩阵的行数和残差的大小有关
        int J_rows = 2 * residualNUM;
        /// 雅克比矩阵的大小
        J.resize(J_rows, J_cols);
        J.setZero();
        /// 残差大小
        fx.resize(J_rows, 1);
        fx.setZero();
        /// 开始求解雅克比
        int J_row_now = 0;
        int fx_row_now = 0;
        ///置零
        fixedPoseNow = 0;
        fixedPointNow = 0;
        for (int currentPoseNum = 0;currentPoseNum<Pose_costFunction.size();++currentPoseNum) {
            Eigen::MatrixXd jacobian;
            Eigen::MatrixXd residual;
            auto currentPose = ID2pose[currentPoseNum];
            auto costFunctions = Pose_costFunction[currentPose];
            //auto t = i.first;
            findJacobian(currentPose, costFunctions, jacobian, residual);
            int j_rows = (int) jacobian.rows();
            int j_cols = (int) jacobian.cols();
            ///如果point 和 pose 全部被约束，那么跳过
            if (j_rows == 0 || j_cols == 0) {
                cout << YELLOW << "warning：jacobian is empty!:Problem::solveByGN" << endl;
                continue;
            }
            ///检测是否有错误数据
            {
                if (J_cols != j_cols) cout << RED << "ERROE:j_cols,Problem::solveByGN" << endl;
                if (j_rows != costFunctions.size() * 2) cout << RED << "ERROE:j_rows,Problem::solveByGN" << endl;
            }
            J.block(J_row_now, 0, j_rows, j_cols) = jacobian;
            fx.block(fx_row_now, 0, residual.rows(), 1) = residual;
            ///更新下一次赋值的位置
            J_row_now += j_rows;
            fx_row_now += residual.rows();
            /// 更新固定变量数目
            if (fixPoseFlag[currentPose]) ++fixedPoseNow;
        }
        //cout<<J<<endl;
        showMatrixAsImage(J);
#ifdef DEBUG
        std::cout << YELLOW << "--------------2.求解雅克比的转职矩阵----------" << RESET << std::endl;
#endif
        Eigen::MatrixXd J_T = J.transpose();
        //showMatrixAsImage(J_T);
#ifdef DEBUG
        std::cout << YELLOW << "----------------3.构造H矩阵-----------------" << RESET << std::endl;
#endif
        Eigen::MatrixXd H = J_T*J;
        //cout<<H<<endl;
        /*
        double min = 9999999.0;
        for(int z=0;z<H.rows();++z){
            if(H(z,z) < min) min = H(z,z);
        }
        cout<<"min element is :"<<min<<endl;
         */
        showMatrixAsImage(H);
#ifdef DEBUG
        std::cout << YELLOW << "----------------4.求解g------------------" << RESET << std::endl;
#endif
        Eigen::VectorXd g = -1 * J_T * fx;
#ifdef DEBUG
        std::cout << YELLOW << "-------------5.求解H的逆--------------" << RESET << std::endl;
#endif
        ///展示矩阵大小
        /// Using the QR decomposition
        auto start = cv::getTickCount();
        Eigen::MatrixXd delt_x = H.colPivHouseholderQr().solve(g);
        /// Schur

        double time = (cv::getTickCount() - start) / cv::getTickFrequency();
#ifdef DEBUG
        std::cout << GREEN << "Iter："<<it;
        std::cout << GREEN << ",Solve Finish ......Cost:" << time << RESET << std::endl;
#endif
#ifdef DEBUG
        std::cout << YELLOW << "-------------6.更新状态量--------------" << RESET << std::endl;
#endif
        /// 更新
        int p = 0;
        //delt_x = 0.0005 * delt_x;
        /// 首先更新位姿
        for(int i=0;i<countPose;++i){
            auto t = ID2pose[i];
            if(fixPoseFlag[t]) continue;
            upDate(t,delt_x.block(p,0,6,1));
            p += 6;
        }
        /// 其次更新特征点
        for(int i=0;i<countPoint;++i){
            auto t = ID2point[i];
            if(fixPointFlag[t]) continue;
            upDate(t,delt_x.block(p,0,3,1));
            p += 3;
        }
#ifdef DEBUG
        std::cout << YELLOW << "-------------7.判断是否收敛--------------" << RESET << std::endl;
#endif
        /// 根据costFunction的值判断是否收敛
        currentCost = 0.0;
        for (int j = 0; j < fx.rows(); ++j) {
            //cout<<"fx:"<<fx(j, 0)<<endl;
            currentCost += fx(j, 0)*fx(j, 0);
        }
#ifdef DEBUG
        std::cout << GREEN << "lastCost：" << lastCost << RESET << std::endl;
        std::cout << GREEN << "currentCost：" << currentCost << RESET << std::endl;
#endif
        if(abs(currentCost - lastCost) > 1.0 && it==MAX_LOOP-1) {
            std::cout << RED << "Bad Solution:The algorithm has been divergent!" << RESET << std::endl;
            std::cout << RED << "currentCost： " << currentCost<<", " ;
            std::cout << RED << "lastCost： " << lastCost << RESET << std::endl;
        }
        else if (abs(currentCost - lastCost)  < 1.0) {
            double costTime = (cv::getTickCount() - BAstart) / cv::getTickFrequency();
            std::cout << GREEN << "The algorithm has converged! ";
            std::cout << BLUE << "Iter Time: "<<it<<", ";
            std::cout << YELLOW << "The Whole algorithm cost: "<<costTime<<", ";
            std::cout << MAGENTA << "currentCost： " << currentCost<<", " ;
            std::cout << MAGENTA << "lastCost： " << lastCost << RESET << std::endl;
            break;
        } else {
            lastCost = currentCost;
        }
    }
}

bool Problem::solveByLM(){
    ///开始计时
    auto BAstart = cv::getTickCount();
    /// 迭代参数重置
    miu = 1.0;
    v = 2.0;
    tao = 1e-6;
    /// 保存Cost
    double currentCost = 0.;
    double lastCost = 0.;
    double newRes;
#ifdef DEBUG
    std::cout<<"位姿数量："<< Pose_costFunction.size()<<std::endl;
    std::cout<<"特征数量："<< ID2point.size()<<std::endl;
#endif
#ifdef DEBUG
    std::cout<<YELLOW<<"-------------------------------0.构造状态量 x----------------------------------"<< RESET<<std::endl;
#endif
    Eigen::MatrixXd x;
    ///优化变量的大小
    flexiblePoseNum  = (int)ID2pose.size() - (int)fixPoseFlag.size();
    flexiblePointNum = (int)ID2point.size() - (int)fixPointFlag.size();

    if(flexiblePointNum == 0) std::cout<<GREEN<<"只优化关键帧位置..."<<RESET<<std::endl;
    if(flexiblePoseNum == 0) std::cout<<GREEN<<"只优化特征点位置..."<<RESET<<std::endl;
    if(flexiblePoseNum == 0 && flexiblePointNum==0) {
        std::cout<<RED<<"错误：优化变量为0..."<<RESET<<std::endl;
        return false;
    }
#ifdef DEBUG
    std::cout << YELLOW << "------------"<<GREEN<<"开始迭代～～"<<BLUE<<"-------------" << RESET << std::endl;
#endif
    /// 估算一下大概有多少个零元素，可以精确计算，但是没有算
    int unZeroBlockSize = 0;
    for(int i = 0;i<Pose_costFunction.size();++i){
        /// 每一帧有多少个观测量
        auto& currentPose = ID2pose[i];
        unZeroBlockSize += Pose_costFunction[currentPose].size();
    }

    for(int it =0;it<MAX_LOOP;++it) {
#ifdef DEBUG
        std::cout << YELLOW << "------------1.构造雅克比矩阵,求解f(x)-------------" << RESET << std::endl;
#endif
        auto itstart = cv::getTickCount();
        /// 雅克比矩阵的行数和残差的大小有关
        int J_rows = 2 * residualNUM;
        /// 雅克比矩阵的列数和优化变量的维度有关
        int J_cols = 6 * flexiblePoseNum + 3 * flexiblePointNum;
        if (J_cols == 0) {
            cout << RED << "ERROR:Problem::solveByGN..." << "J_cols=0" << endl;
            break;
        }
        ///置零
        fixedPoseNow = 0;
        fixedPointNow = 0;
        ///
        vector<TRI> jacobian;
        jacobian.reserve(unZeroBlockSize*16 + 2000);
        vector<double> residual;
        residual.reserve(unZeroBlockSize*2 + 500);
        int rowNow =0;
        for (int currentPoseNum = 0;currentPoseNum<Pose_costFunction.size();++currentPoseNum) {
            auto& currentPose = ID2pose[currentPoseNum];
            auto& costFunctions = Pose_costFunction[currentPose];
            findSparseJacobian(currentPose,costFunctions,jacobian,residual,rowNow);
            /*
            int j_rows = (int) jacobian.rows();
            int j_cols = (int) jacobian.cols();
            ///如果point 和 pose 全部被约束，那么跳过
            if (j_rows == 0 || j_cols == 0) {
                cout << YELLOW << "warning：jacobian is empty!:Problem::solveByLM" << endl;
                continue;
            }
            ///检测是否有错误数据
            {
                if (J_cols != j_cols) cout << RED << "ERROE:j_cols,Problem::solveByGN" << endl;
                if (j_rows != costFunctions.size() * 2) cout << RED << "ERROE:j_rows,Problem::solveByLM" << endl;
            }
            J.block(J_row_now, 0, j_rows, j_cols) = jacobian;
            fx.block(fx_row_now, 0, residual.rows(), 1) = residual;
            ///更新下一次赋值的位置
            J_row_now += j_rows;
            fx_row_now += residual.rows();
             */
            /// 更新固定变量数目
            if (fixPoseFlag[currentPose]) ++fixedPoseNow;
        }
        /// 把j转换为稀疏矩阵
        Eigen::MatrixXd fx(residual.size(),1);
        for(int i=0;i<residual.size();++i){
            fx(i,0) = residual[i];
        }
        //cout<<"jacobian size:"<<jacobian.size()<<endl;
        SM SJ(J_rows,J_cols);
        SJ.setFromTriplets(jacobian.begin(),jacobian.end());
        //double ti = (cv::getTickCount() - start) / cv::getTickFrequency();
        //std::cout << GREEN << "tri ......Cost:" << ti << RESET << std::endl;
        SJ.makeCompressed();
        ///
#ifdef DEBUG
        std::cout << YELLOW << "--------------2.求解雅克比的转职矩阵----------" << RESET << std::endl;
#endif
        //Eigen::MatrixXd J_T = J.transpose();
        SM SJ_T = SJ.transpose();
        //showMatrixAsImage(J_T);
#ifdef DEBUG
        std::cout << YELLOW << "----------------3.构造H矩阵-----------------" << RESET << std::endl;
#endif
        //SM SH = SJ_T*SJ;
        /// pruned耗费时间好长
        //SM SH = (SJ_T*SJ).pruned(1e-5);
        SM SH = SJ_T*SJ;
        //cout<<"H rows:"<<SH.rows()<<endl;
        //cout<<"H cols:"<<SH.cols()<<endl;
        //Eigen::MatrixXd H = J_T * J;
        int LMCount = 3;
        while(LMCount) {
            --LMCount;
            if (it == 0) {
                miu = std::numeric_limits<double>::min();
                for (int row = 0; row < SH.rows(); ++row) {
                    if (SH.coeffRef(row,row) > miu) miu = SH.coeffRef(row,row);
                }
                miu *= tao;
            }
            /// H + miu*I

            for (int z = 0; z < SH.rows(); ++z) {
                SH.coeffRef(z,z) += miu;
            }
            //cout<<"miu:"<<miu<<endl;
            SH.makeCompressed();
            showMatrixAsImage(SH);
#ifdef DEBUG
            std::cout << YELLOW << "----------------4.求解g------------------" << RESET << std::endl;
#endif
            //Eigen::VectorXd g = -1 * J_T * fx;
            Eigen::VectorXd g = -1.0 * SJ_T * fx;
#ifdef DEBUG
            std::cout << YELLOW << "-------------5.求解H的逆--------------" << RESET << std::endl;
#endif
            ///展示矩阵大小
            /// Using the QR decomposition
            //Eigen::MatrixXd delt_x = H.colPivHouseholderQr().solve(g);
            /// Schur
            //auto start = cv::getTickCount();

            Eigen::MatrixXd delt_x = solveHByFactorization(SH,g);
            //Eigen::MatrixXd delt_x = solveHBySolveBlock(SH,g);

            //cout<<"rows:"<<delt_x.rows()<<endl;
            //cout<<"cols:"<<delt_x.cols()<<endl;
            //double time = (cv::getTickCount() - start) / cv::getTickFrequency();
            //std::cout << GREEN << "C_inverse ......Cost:" << time << RESET << std::endl;
            //double sumDeltX = (delt_x.transpose()*delt_x)(0,0);
            //cout<<"sumDeltX:"<<sumDeltX<<endl;

            //if(sumDeltX<1e-6) break;
#ifdef DEBUG
            std::cout << GREEN << "Iter：" << it;
            std::cout << GREEN << ",Solve Finish ......Cost:" << time << RESET << std::endl;
#endif
#ifdef DEBUG
            std::cout << YELLOW << "-------------6.更新状态向量--------------" << RESET << std::endl;
#endif
            /// 更新
            int p = 0;
            /// 首先更新位姿
            for (int i = 0; i < countPose; ++i) {
                auto t = ID2pose[i];
                if (fixPoseFlag[t]) continue;
                /// 记录旧变量，方便回滚
                ID2oldPose[i] = *t;
                upDate(t, delt_x.block(p, 0, 6, 1));
                p += 6;
            }
            /// 其次更新特征点
            for (int i = 0; i < countPoint; ++i) {
                auto t = ID2point[i];
                if (fixPointFlag[t]) continue;
                ID2oldPoint[i] = *t;
                upDate(t, delt_x.block(p, 0, 3, 1));
                p += 3;
            }
#ifdef DEBUG
            std::cout << YELLOW << "-------------7.计算μ--------------" << RESET << std::endl;
#endif
            ///判断近似的情况
            double approximationRes = (delt_x.transpose() * (miu * delt_x + g))(0, 0);
            approximationRes *= 0.5;
            newRes = 0.0;
            /// 计算更新后的残差
            for (int currentPoseNum = 0; currentPoseNum < Pose_costFunction.size(); ++currentPoseNum) {
                auto& currentPose = ID2pose[currentPoseNum];
                auto& costFunctions = Pose_costFunction[currentPose];
                /// 计算残差
                for (int i = 0; i < costFunctions.size(); i++) {
                    //Block of size (p,q), starting at (i,j)
                    //matrix.block(i,j,p,q)->针对动态矩阵,matrix.block<p,q>(i,j)->针对静态矩阵
                    Eigen::Matrix<double, 2, 1> residualBlock;
                    costFunctions[i]->getResidual(residualBlock);

                    newRes += residualBlock(0, 0) * residualBlock(0, 0);
                    newRes += residualBlock(1, 0) * residualBlock(1, 0);
                }
            }
            /// 计算旧残差
            double oldRes = (fx.transpose() * fx)(0, 0);
            /// 计算ρ
            double rou = (oldRes - newRes) / approximationRes;
            if (rou > 0.0) {
                // µ := µ ∗ max{ 1/3 , 1 − (2% − 1)3}; ν := 2
                miu = miu * max<double>(0.333, 1.0 - pow((2.0 * rou) - 1.0, 3));
                v = 2.0;
                break;
            } else {
                miu *= v;
                v *= 2.0;
                rollBack();
            }
            /*
            if(LMCount==0 && rou<0.0){
                return true;
            }
            */
            //cout << "ρ:" << rou << endl;
            //std::cout << GREEN << "          oldRes：" << oldRes << RESET << std::endl;
            //std::cout << GREEN << "          newRes：" << newRes << RESET << std::endl;
            //std::cout << GREEN << "approximationRES：" << approximationRes << RESET << std::endl;
            //std::cout << GREEN << "         realRES：" << oldRes - newRes << RESET << std::endl;
        }
#ifdef DEBUG
        std::cout << YELLOW << "-------------8.判断是否收敛--------------" << RESET << std::endl;
#endif
        /// 根据costFunction的值判断是否收敛
        currentCost = (fx.transpose()*fx)(0,0);
        //currentCost = newRes;
#ifdef DEBUG
        std::cout << GREEN << "lastCost：" << lastCost << RESET << std::endl;
        std::cout << GREEN << "currentCost：" << currentCost << RESET << std::endl;
#endif
        double tcostTime = (cv::getTickCount() - itstart) / cv::getTickFrequency();
        std::cout<< "这一轮循环共花费: "<<tcostTime<<endl;

        if(abs(lastCost - currentCost) > 5.0 && it==MAX_LOOP-1) {
            std::cout << RED << "Bad Solution:The algorithm has been divergent!" << RESET << std::endl;
            std::cout << RED << "currentCost： " << currentCost<<", " ;
            std::cout << RED << "lastCost： " << lastCost << RESET << std::endl;
        }
        else if (abs(lastCost - currentCost)  < 5.0) {
            double costTime = (cv::getTickCount() - BAstart) / cv::getTickFrequency();
            std::cout << GREEN << "The algorithm has converged! ";
            std::cout << BLUE << "Iter Time: "<<it<<", ";
            std::cout << YELLOW << "The Whole algorithm cost: "<<costTime<<", ";
            std::cout << MAGENTA << "currentCost： " << currentCost<<", " ;
            std::cout << MAGENTA << "lastCost： " << lastCost << RESET << std::endl;
            break;
        } else {
            lastCost = currentCost;
        }
        //double tcostTime = (cv::getTickCount() - itstart) / cv::getTickFrequency();
        //std::cout<< "这一轮循环共花费: "<<tcostTime<<endl;
    }
    return true;
}
void Problem::rollBack() {
    ///恢复姿态
    for(int i=0;i<countPose;++i){
        auto t = ID2pose[i];
        if(fixPoseFlag[t]) continue;
        *t = ID2oldPose[i];
    }
    /// 其次恢复特征点
    for(int i=0;i<countPoint;++i){
        auto t = ID2point[i];
        if(fixPointFlag[t]) continue;
        *t = ID2oldPoint[i];
    }
}
void Problem::upDate(ptrPose x,Eigen::Matrix<double ,6,1> delt_x){
    /*
    ///李代数相加需要添加一些余项，转化为R再相乘，代替加法,详见14讲 72页；
    ///把SE3上的李代数转化为4x4矩阵
    Eigen::Matrix4d Pos_Matrix = Sophus::SE3::exp(*x).matrix();
    Eigen::Matrix4d Pos_update_Matrix = Sophus::SE3::exp(delt_x).matrix();
    ///矩阵更新
    Pos_Matrix = Pos_update_Matrix*Pos_Matrix;
    ///转化为李代数
    Sophus::SE3 new_Pos_se = Sophus::SE3(Pos_Matrix.block<3, 3>(0, 0), Pos_Matrix.block<3, 1>(0, 3));
    ///更新姿态
    *x= new_Pos_se.log();
     */
    Sophus::SE3 delta_SE3 = Sophus::SE3::exp(delt_x);
    Sophus::SE3 pose = Sophus::SE3::exp(*x);
    Sophus::SE3 new_Pos_se = delta_SE3*pose;
    *x= new_Pos_se.log();
}
void Problem::upDate(ptr3D x,Eigen::Vector3d delt_x){
    /// 特征点坐标直接相加
    *x = *x + delt_x;
}
Eigen::MatrixXd Problem::solveHByFactorization(SM& H,Eigen::VectorXd& g){
    /// 使用Eigen求逆
    /// Schur消元
    /*
    SolverClassName<SparseMatrix<double> > solver;
    solver.compute(A);
    SparseMatrix<double> I(n,n);
    I.setIdentity();
    auto A_inv = solver.solve(I);
     */

    Eigen::MatrixXd ans;
    /// number of camera
    /// number of point
    const int& NC = flexiblePoseNum;
    const int& NP = flexiblePointNum;
    ans.resize(NC*6+NP*3,1);

    //SM B = H.topLeftCorner(NC*6,NC*6);
    //SM E = H.topRightCorner(NC*6,NP*3);
    //SM E_T = H.bottomLeftCorner(NP*3,NC*6);
    //SM C = H.bottomRightCorner(NP*3,NP*3);
    /// QR
    //auto start = cv::getTickCount();
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QRsolver;
    if(NC==0 ||NP==0){
        QRsolver.compute(H);
        ans = QRsolver.solve(g);
        return ans;
    }
    // QRsolver.compute(H.bottomRightCorner(NP*3,NP*3));
    // SM I(NP*3,NP*3);
    // I.setIdentity();
    // SM C_inverse = QRsolver.solve(I);
    // double time = (cv::getTickCount() - start) / cv::getTickFrequency();
    // std::cout << GREEN << "C_inverse ......Cost:" << time << RESET << std::endl;

    /// LDLT
    //SM C = H.bottomRightCorner(NP*3,NP*3);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLTsolver;
    LDLTsolver.compute(H.bottomRightCorner(NP*3,NP*3));
    SM I(NP*3,NP*3);
    I.setIdentity();
    SM C_inverse = LDLTsolver.solve(I);
    //std::cout << GREEN << "C_inverse ......Cost:" << time << RESET << std::endl;
    /// LU
    /*
    auto start = cv::getTickCount();
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> LUsolver;
    LUsolver.compute(H.bottomRightCorner(NP*3,NP*3));
    SM I(NP*3,NP*3);
    I.setIdentity();
    SM C_inverse = LUsolver.solve(I);
    double time = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << GREEN << "C_inverse ......Cost:" << time << RESET << std::endl;
    */
    //Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QRsolverCamera;
    //SM tempH = H.topLeftCorner(NC*6,NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * H.bottomLeftCorner(NP*3,NC*6);
    //QRsolverCamera.compute(H.topLeftCorner(NC*6,NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * H.bottomLeftCorner(NP*3,NC*6));
    QRsolver.compute(H.topLeftCorner(NC*6,NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * H.bottomLeftCorner(NP*3,NC*6));
    //Eigen::VectorXd tempG = g.topRows(NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * g.bottomRows(NP*3);
    ans.topRows(NC*6) = QRsolver.solve(g.topRows(NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * g.bottomRows(NP*3));
    ans.bottomRows(NP*3) = C_inverse*(g.bottomRows(NP*3) - H.bottomLeftCorner(NP*3,NC*6)*ans.topRows(NC*6));
    //double time = (cv::getTickCount() - start) / cv::getTickFrequency();
    //std::cout << GREEN << "solve ......Cost:" << time << RESET << std::endl;
    /*
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QRsolver;
    QRsolver.compute(H);
    Eigen::MatrixXd ans = QRsolver.solve(g);
    //cout<<ans<<endl;
    */
    return ans;
}

Eigen::MatrixXd Problem::solveHBySolveBlock(SM& H,Eigen::VectorXd& vectorg){
    /// 手动对C矩阵求逆
    ///
    Eigen::MatrixXd ans;
    /// number of camera
    /// number of point
    const int& NC = flexiblePoseNum;
    const int& NP = flexiblePointNum;
    ans.resize(NC*6+NP*3,1);

    //SM B = H.topLeftCorner(NC*6,NC*6);
    //SM E = H.topRightCorner(NC*6,NP*3);
    //SM E_T = H.bottomLeftCorner(NP*3,NC*6);
    //SM C = H.bottomRightCorner(NP*3,NP*3);

    SM C = H.bottomRightCorner(NP*3,NP*3);
    SM C_inverse(NP*3,NP*3);
    int pointNow;
    int rowNow;
    int colNow;
    double a,b,c,d,e,f,g,h,i;
    /// 首先声明存储空间
    //C_inverse.reserve(NP*9);
    //auto start = cv::getTickCount();
    for(pointNow = 0;pointNow<NP;++pointNow){

        rowNow = pointNow*3;
        colNow = pointNow*3;

        a = C.coeffRef(rowNow,colNow);
        b = C.coeffRef(rowNow,colNow + 1);
        c = C.coeffRef(rowNow,colNow + 2);

        d = C.coeffRef(rowNow + 1,colNow);
        e = C.coeffRef(rowNow + 1,colNow + 1);
        f = C.coeffRef(rowNow + 1,colNow + 2);

        g = C.coeffRef(rowNow + 2,colNow);
        h = C.coeffRef(rowNow + 2,colNow + 1);
        i = C.coeffRef(rowNow + 2,colNow + 2);

        /// 首先求解行列式
        double detA = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h;
        detA = 1.0/detA;

        C_inverse.insert(rowNow    ,colNow    ) = detA*(e*i - h*f);
        C_inverse.insert(rowNow    ,colNow + 1) = detA*(-(b*i-h*c));
        C_inverse.insert(rowNow    ,colNow + 2) = detA*(b*f-c*e);

        C_inverse.insert(rowNow + 1,colNow    ) = detA*(f*g - i*d);
        C_inverse.insert(rowNow + 1,colNow + 1) = detA*(-1.0*(c*g - i*a));
        C_inverse.insert(rowNow + 1,colNow + 2) = detA*(c*d - a*f);

        C_inverse.insert(rowNow + 2,colNow    ) = detA*(d*h - g*e);
        C_inverse.insert(rowNow + 2,colNow + 1) = detA*(-1.0*(a*h - g*b));
        C_inverse.insert(rowNow + 2,colNow + 2) = detA*(a*e - h*d);

        /*
        C_inverse.coeffRef(rowNow    ,colNow    ) = detA*(e*i - h*f);
        C_inverse.coeffRef(rowNow    ,colNow + 1) = detA*(-1.0*(b*i-h*c));
        C_inverse.coeffRef(rowNow    ,colNow + 2) = detA*(b*f-c*e);

        C_inverse.coeffRef(rowNow + 1,colNow    ) = detA*(f*g - i*d);
        C_inverse.coeffRef(rowNow + 1,colNow + 1) = detA*(-1.0*(c*g - i*a));
        C_inverse.coeffRef(rowNow + 1,colNow + 2) = detA*(c*d - a*f);

        C_inverse.coeffRef(rowNow + 2,colNow    ) = detA*(d*h - g*e);
        C_inverse.coeffRef(rowNow + 2,colNow + 1) = detA*(-1.0*(a*h - g*b));
        C_inverse.coeffRef(rowNow + 2,colNow + 2) = detA*(a*e - h*d);
         */
    }
    //double time = (cv::getTickCount() - start) / cv::getTickFrequency();
    //std::cout  << "C_inverse ......Cost:" << time << RESET << std::endl;
    /// QR
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QRsolver;
    /// LU
    //Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QRsolver;
    /// LDLT
    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> QRsolver;
    QRsolver.compute(H.topLeftCorner(NC*6,NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * H.bottomLeftCorner(NP*3,NC*6));
    ans.topRows(NC*6) = QRsolver.solve(vectorg.topRows(NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * vectorg.bottomRows(NP*3));
    ans.bottomRows(NP*3) = C_inverse*(vectorg.bottomRows(NP*3) - H.bottomLeftCorner(NP*3,NC*6)*ans.topRows(NC*6));
    return ans;
}
void Problem::showMatrixAsImage(Eigen::MatrixXd& input,string windowName){
#ifdef SHOWMATRIX
    int rows = (int)input.rows();
    int cols = (int)input.cols();
    std::cout<<GREEN<<"Matrix size:"<<rows<<"x"<<cols<<RESET<<endl;
    cv::Mat Matrix(rows, cols, CV_8UC1, cv::Scalar(255));
    for (int i = 0; i <rows; ++i)
    {
        /// image.ptr<>(i)返回第i行首元素的指针
        auto p = Matrix.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j)
        {
            if(input(i,j)!=0) p[j] = 0;
        }
    }
    cv::imshow(windowName,Matrix);
    cv::waitKey(0);
#endif
}
void Problem::showMatrixAsImage(SM& input,string windowName){
#ifdef SHOWMATRIX
    int rows = (int)input.rows();
    int cols = (int)input.cols();
    //std::cout<<GREEN<<"Matrix size:"<<rows<<"x"<<cols<<RESET<<endl;
    if(rows<2000) return;
    std::cout<<"Showing......"<<std::endl;
    std::cout<<GREEN<<"Matrix size:"<<rows<<"x"<<cols<<RESET<<endl;
    cv::Mat Matrix(rows, cols, CV_8UC1, cv::Scalar(255));
    for (int i = 0; i <rows; ++i)
    {
        /// image.ptr<>(i)返回第i行首元素的指针
        auto p = Matrix.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j)
        {
            if(input.coeffRef(i,j)!=0.0) p[j] = 0;
        }
    }
    cv::imshow(windowName,Matrix);
    cv::waitKey(0);
#endif
}
