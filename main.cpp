#include <iostream>
#include "env.h"
#include "J_layers.h"
#include "J_kf.h"

using namespace std;

const double v_x[4] = {-0.3, 0.3, 0, 0};
const double v_y[4] = {0, 0, 0.3, -0.3};

struct DQN {
	J_liner_layer layer1;
	J_tanh_layer layer2;
	J_liner_layer layer3;
	J_relu_layer layer4;
	J_liner_layer layer5;
	J_sigmoid_layer layer6;

	DQN() : layer1(4, 10, 0.01, 1, 0.2, 0.999, 0.001, 0), layer2(10), layer3(10, 10, 0.01, 1, 0.2, 0.999, 0.001, 1), layer4(10), layer5(10, 4, 0.01, 1, 0.2, 0.999, 0.001, 1), layer6(4) {}
	vector<double> forward(vector<double> input) {
		input = layer1.forward(input);
		input = layer2.forward(input);
		input = layer3.forward(input);
		input = layer4.forward(input);
		input = layer5.forward(input);
		input = layer6.forward(input);
		return input;
	}
	void backward(vector<double> loss) {
		loss = layer6.backward(loss);
		loss = layer5.backward(loss);
		loss = layer4.backward(loss);
		loss = layer3.backward(loss);
		loss = layer2.backward(loss);
		loss = layer1.backward(loss);
	}
	void save(string path = "test") {
		layer1.save(path + "layer1");
		layer3.save(path + "layer3");
		layer5.save(path + "layer5");
	}
	void load(string path = "test") {
		layer1.load(path + "layer1");
		layer3.load(path + "layer3");
		layer5.load(path + "layer5");
	}
};

struct MLP {
	J_liner_layer layer1;
	J_tanh_layer layer2;
	J_liner_layer layer3;
	J_sigmoid_layer layer4;

	MLP() : layer1(10, 20, 0.01, 1, 0.2, 0.999, 0.001, 0), layer2(20), layer3(20, 1, 0.01, 1, 0.2, 0.999, 0.001, 1), layer4(1) {}
	vector<double> forward(vector<double> input) {
		input = layer1.forward(input);
		input = layer2.forward(input);
		input = layer3.forward(input);
		input = layer4.forward(input);
		return input;
	}

	void backward(vector<double> loss) {
		loss = layer4.backward(loss);
		loss = layer3.backward(loss);
		loss = layer2.backward(loss);
		loss = layer1.backward(loss);
	}

	void save(string path = "test") {
		layer1.save(path + "layer1");
		layer3.save(path + "layer3");
	}
	void load(string path = "test") {
		layer1.load(path + "layer1");
		layer3.load(path + "layer3");
	}
};

struct System {
	DQN dqn;
	MLP mlp0, mlp1, mlp2, mlp;
	int last_action;
	int count;
	Eigen::Matrix2d covP_x; // 协方差
	Eigen::Vector2d State_x; // 状态
	Eigen::Matrix2d covP_y;
	Eigen::Vector2d State_y;
	vector<double> memory_bias;

	System() : dqn(), mlp0(), mlp1(), mlp2(), mlp(), last_action(0), count(0), covP_x(Eigen::Matrix2d::Identity()), State_x(Eigen::Vector2d::Zero()), covP_y(Eigen::Matrix2d::Identity()), State_y(Eigen::Vector2d::Zero()) {}
	int reset(vector<double> state) {
		covP_x << 1, 0, 0, 1; //协方差一开始为1，后续会迭代收敛
		State_x << state[0], 0; // 初始状态值， 第一个为位移、第二个为速度
		covP_y << 1, 0, 0, 1; //协方差一开始为1，后续会迭代收敛
		State_y << state[1], 0; // 初始状态值， 第一个为位移、第二个为速度
		memory_bias = { 0,0 };
		count = 0;
		vector<double> q_values = dqn.forward(state);
		last_action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
		return last_action;
	}

	int act(vector<double> state) {
		kf(state[0], v_x[last_action], covP_x, State_x);
		kf(state[1], v_y[last_action], covP_y, State_y);
		memory_bias.push_back(state[0] - State_x[0]);
		memory_bias.push_back(state[1] - State_y[0]);
		if(memory_bias.size() > 10) {
			memory_bias.erase(memory_bias.begin());
			memory_bias.erase(memory_bias.begin());
			if(mlp0.forward(memory_bias)[0] > 0.5 || mlp1.forward(memory_bias)[0] > 0.5 || mlp2.forward(memory_bias)[0] > 0.5 || mlp.forward(memory_bias)[0] > 0.5) {
				count++;
			}
			else {
				count = 0;
			}
			if (count >= 3) {
				return -1;
			}
		}
		state[0] = State_x[0];
		state[1] = State_y[0];
		vector<double> q_values = dqn.forward(state);
		last_action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
		return last_action;
	}

	void load() {
		dqn.load("trained models/dqn ");
		mlp.load("trained models/mlp ");
		mlp0.load("trained models/mlp0 ");
		mlp1.load("trained models/mlp1 ");
		mlp2.load("trained models/mlp2 ");
	}

	void save() {
		dqn.save("trained models/dqn ");
		mlp.save("trained models/mlp ");
		mlp0.save("trained models/mlp0 ");
		mlp1.save("trained models/mlp1 ");
		mlp2.save("trained models/mlp2 ");
	}
};

void pure_dqn_visual() {
	Env env;
	DQN dqn;
	dqn.load("trained models/dqn ");
	initgraph(300, 300);
	env.reset(10);
	env.print_bkg();
	for(int step = 0; step < 1000; step++) {
		vector<double> state = env.observe();
		vector<double> q_values = dqn.forward(state);
		int action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
		env.step(action);
		env.print_obs_route();
		env.print_drone_route(50);
		if(env.is_done()) {
			break;
		}
	}
	
}

void pure_dqn() {
	cout << "pure_dqn:" << endl;
	Env env;
	DQN dqn;
	dqn.load("trained models/dqn ");
	double avg_pos = 0;
	int success = 0;
	int break_count = 0;
	int success_break = 0;
	int good_stop = 0;
	int bad_stop = 0;
	int crash = 0;
	for (int episode = 0; episode < 10000; episode++) {
		env.reset(10);
		for (int step = 0; step < 1000; step++) {
			vector<double> state = env.observe();
			vector<double> q_values = dqn.forward(state);
			int action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
			if (action == -1) {
				if (env.is_break_down()) {
					break_count++;
					good_stop++;
				}
				else {
					bad_stop++;
				}
				break;
			}
			env.step(action);
			if (env.is_done()) {
				avg_pos += env.get_pos();
				if (env.get_pos() == 3) {
					success++;
				}
				else {
					crash++;
				}
				if (env.is_break_down()) {
					break_count++;
				}
				break;
			}
		}
	}
	cout << "escape: " << success << " / 10000" << endl;
	cout << "crash: " << crash << " /10000" << endl;
	cout << "mis-stop: " << bad_stop << " /10000" << endl;
	cout << "reasonable stop: " << good_stop << " / " << break_count << endl;
}

void kf_dqn_visual() {
	Env env;
	System system;
	system.load();
	initgraph(300, 300);
	env.reset(10);
	env.print_bkg();
	vector<double> state = env.observe();
	int action = system.reset(state);
	env.step(action);
	for(int step = 0; step < 1000; step++) {
		vector<double> state = env.observe();
		int action = system.act(state);
		env.step(action);
		env.print_obs_route();
		env.print_drone_route(50);
		if(env.is_done()) {
			break;
		}
	}
}

void kf_dqn() {
	cout << "trained_system:" << endl;
	Env env;
	System system;
	system.load();
	double avg_pos = 0;
	int success = 0;
	int break_count = 0;
	int success_break = 0;
	int good_stop = 0;
	int bad_stop = 0;
	int crash = 0;
	for (int episode = 0; episode < 10000; episode++) {
		env.reset(10);
		vector<double> state = env.observe();
		int action = system.reset(state);
		env.step(action);
		for (int step = 0; step < 1000; step++) {
			vector<double> state = env.observe();
			int action = system.act(state);
			if (action == -1) {
				if (env.is_break_down()) {
					break_count++;
					good_stop++;
				}
				else {
					bad_stop++;
				}
				break;
			}
			env.step(action);
			if (env.is_done()) {
				avg_pos += env.get_pos();
				if(env.get_pos() == 3) {
					success++;
				}
				else {
					crash++;
				}
				if(env.is_break_down()) {
					break_count++;
				}
				break;
			}
		}
	}
	cout << "escape: " << success << " / 10000" << endl;
	cout << "crash: " << crash << " /10000" << endl;
	cout << "mis-stop: " << bad_stop << " /10000" << endl;
	cout << "reasonable stop: " << good_stop << " / " << break_count << endl;
}

int main() {
	while(1)
	{
		cout << "** 0.pure_dqn_visual    1.trained_system_visual    2.pure_dqn   3.trained_system **" << endl;
		cout << "please input the number of the function you want to run:";
		int op;
		cin >> op;
		switch (op) {
			case 0:
				pure_dqn_visual();
				break;
			case 1:
				kf_dqn_visual();
				break;
			case 2:
				pure_dqn();
				break;
			case 3:
				kf_dqn();
				break;
			default:
				return 0;
		}
		cout << endl;
		cout << "*******************************************************************" << endl;
	}
	return 0;
}