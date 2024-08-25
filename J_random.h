#pragma once
#include <random>
#include <vector>
int int_random(int a = 0, int b = 100) {
	if (a > b) {
		int temp = a;
		a = b;
		b = temp;
	}
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(a, b);
	return dis(gen);
}

double double_random_dis_0(double mean = 0, double stddev = 1, double max = 5) {
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> distribution(mean, stddev);
	while (1) {
		double num = distribution(gen);
		if (num < max && num > -max) {
			return num;
		}
	}
}

std::vector<std::vector<double>> xavier_init(int n_in, int n_out) {
	std::random_device rd;  // 用于获取随机数种子的设备  
	std::mt19937 gen(rd()); // 基于 Mersenne Twister 算法的随机数生成器  
	double a = std::sqrt(6.0 / double(n_in + n_out));
	std::uniform_real_distribution<> dis(-a, a); // 均匀分布  

	std::vector<std::vector<double>> weights(n_out, std::vector<double>(n_in));

	for (int i = 0; i < n_out; ++i) {
		for (int j = 0; j < n_in; ++j) {
			weights[i][j] = dis(gen);
		}
	}

	return weights;
}