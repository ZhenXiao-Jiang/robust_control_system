#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

const Eigen::Matrix2d F = (Eigen::Matrix2d() << 1, 1, 0, 1).finished(); // Ԥ��ģ�͵�F(״̬ת�ƾ���)
//const Eigen::Vector2d B{ 0.5,1 }; // Ԥ��ģ�͵�B                                                                  **unnessary in this case
const Eigen::Matrix2d H = Eigen::Matrix2d::Identity(); // ����ģ�͵�H
const Eigen::Matrix2d Q = (Eigen::Matrix2d() << 1e-5, 0, 0, 1e-5).finished(), R = (Eigen::Matrix2d() << 0.001, 0, 0, 0).finished();

void kf(double detect_pos, double detect_vel, Eigen::Matrix2d& covP, Eigen::Vector2d& State) {
    Eigen::Vector2d z; // ���ù۲�״̬������λ�ơ��ٶ�
    //double detect_vel = detect_pos - State(0, 0); // �۲쵽���ٶ�
    z << detect_pos, detect_vel;
    Eigen::Matrix2d I;
    I.setIdentity(); // ����λ��

    //double cutAcc = detect_vel - State(1, 0); // �۲쵽�ļ��ٶ�                                                        **unnessary in this case

    // Ԥ��
    //State = F * State + B * cutAcc; // ����λ��                                                                             **unnessary in this case
    State = F * State; // ����λ��                          
    covP = F * covP * F.transpose() + Q; // Э����

    // ����
    Eigen::Matrix2d K = covP * H.transpose() * (H * covP * H.transpose() + R).inverse(); // ����������
    State = State + K * (z - H.transpose() * State); // �Ż��������
    covP = (I - K) * covP; // ����Э����
}

/*
    Eigen::Matrix2d covP; // Э����
    Eigen::Vector2d State;

    covP << 1, 0, 0, 1; //Э����һ��ʼΪ1���������������
    State << detect_pos[0], 0; // ��ʼ״ֵ̬�� ��һ��Ϊλ�ơ��ڶ���Ϊ�ٶ�

    kf(detect_pos[i], detect_vel[i], covP, State);
*/