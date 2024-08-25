#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

const Eigen::Matrix2d F = (Eigen::Matrix2d() << 1, 1, 0, 1).finished(); // 预测模型的F(状态转移矩阵)
//const Eigen::Vector2d B{ 0.5,1 }; // 预测模型的B                                                                  **unnessary in this case
const Eigen::Matrix2d H = Eigen::Matrix2d::Identity(); // 测量模型的H
const Eigen::Matrix2d Q = (Eigen::Matrix2d() << 1e-5, 0, 0, 1e-5).finished(), R = (Eigen::Matrix2d() << 0.001, 0, 0, 0).finished();

void kf(double detect_pos, double detect_vel, Eigen::Matrix2d& covP, Eigen::Vector2d& State) {
    Eigen::Vector2d z; // 设置观测状态向量，位移、速度
    //double detect_vel = detect_pos - State(0, 0); // 观察到的速度
    z << detect_pos, detect_vel;
    Eigen::Matrix2d I;
    I.setIdentity(); // 矩阵单位化

    //double cutAcc = detect_vel - State(1, 0); // 观察到的加速度                                                        **unnessary in this case

    // 预测
    //State = F * State + B * cutAcc; // 估计位移                                                                             **unnessary in this case
    State = F * State; // 估计位移                          
    covP = F * covP * F.transpose() + Q; // 协方差

    // 更新
    Eigen::Matrix2d K = covP * H.transpose() * (H * covP * H.transpose() + R).inverse(); // 卡尔曼增益
    State = State + K * (z - H.transpose() * State); // 优化后的数据
    covP = (I - K) * covP; // 更新协方差
}

/*
    Eigen::Matrix2d covP; // 协方差
    Eigen::Vector2d State;

    covP << 1, 0, 0, 1; //协方差一开始为1，后续会迭代收敛
    State << detect_pos[0], 0; // 初始状态值， 第一个为位移、第二个为速度

    kf(detect_pos[i], detect_vel[i], covP, State);
*/