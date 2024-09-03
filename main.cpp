#include <iostream>
#include "env.h"
#include "J_layers.h"
#include "J_kf.h"

using namespace std;

int __filter_flag = 0;

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

struct LSTM {
	double _c, _x, _y;
	double grad_f, grad_c;
	J_liner_layer l_f_1, l_c_1, l_f, l_c;
	J_sigmoid_layer s_f_1, s_c_1, s_f, s_c, s;

	LSTM() {
		l_f = J_liner_layer(10, 1);
		l_c = J_liner_layer(10, 1);
		s_f = J_sigmoid_layer(1);
		s_c = J_sigmoid_layer(1);
		l_f_1 = J_liner_layer(7, 10);
		l_c_1 = J_liner_layer(6, 10);
		s_f_1 = J_sigmoid_layer(10);
		s_c_1 = J_sigmoid_layer(10);
		s = J_sigmoid_layer(6);
		_c = 0;
		_x = 0;
		_y = 0;
		grad_f = 0;
		grad_c = 0;
	}

	void reset(double x, double y) {
		_c = 0;
		_x = x;
		_y = y;
	}

	vector<double> forward(vector<double> input) {
		if (input.size() != 4) {
			cout << "input size error" << endl;
			exit(0);
		}
		input.push_back(_x);
		input.push_back(_y);
		vector<double> sf = s.forward(input);
		sf.push_back(_c);
		double f = s_f.forward(l_f.forward(s_f_1.forward(l_f_1.forward(sf))))[0];
		double c = s_c.forward(l_c.forward(s_c_1.forward(l_c_1.forward(input))))[0];
		grad_f = _c - c;
		grad_c = 1 - f;
		_c = f * _c + (1 - f) * c;
		_x = (_x + input[2]) * _c + input[0] * (1 - _c);
		_y = (_y + input[3]) * _c + input[1] * (1 - _c);
		return { _x, _y };
	}

	void save(string name = "lstm ") {
		l_f.save(name + "l_f");
		l_c.save(name + "l_c");
		l_f_1.save(name + "l_f_1");
		l_c_1.save(name + "l_c_1");
	}

	void load(string name = "lstm ") {
		l_f.load(name + "l_f");
		l_c.load(name + "l_c");
		l_f_1.load(name + "l_f_1");
		l_c_1.load(name + "l_c_1");
	}
};

struct HM {
	DQN dqn;
	int last_action;
	int count;
	double last_x, last_y, step;

	HM() : dqn(), last_action(0), count(0), last_x(0), last_y(0),step(0) {}
	int reset(vector<double> state) {
		last_x = state[0];
		last_y = state[1];
		step = 0;
		count = 0;
		vector<double> q_values = dqn.forward(state);
		last_action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
		return last_action;
	}

	int act(vector<double> state) {
		step++;
		double confidence_rate = 0.9 * (2.0  / (1.0 + exp(-step)) - 1);
		state[0] = (1 - confidence_rate) * state[0] + confidence_rate * (last_x + v_x[last_action]);
		state[1] = (1 - confidence_rate) * state[1] + confidence_rate * (last_y + v_y[last_action]);
		last_x = state[0];
		last_y = state[1];
		vector<double> q_values = dqn.forward(state);
		last_action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
		return last_action;
	}

	vector<double> get_state() {
		return { last_x, last_y };
	}

	void load() {
		dqn.load("trained models/dqn ");
	}

	void save() {
		dqn.save("trained models/dqn ");
	}
};

struct System {
	DQN dqn;
	MLP mlp0, mlp1, mlp2, mlp;
	LSTM lstm, lstm1, lstm2;
	int last_action;
	int count;
	Eigen::Matrix2d covP_x; // 协方差
	Eigen::Vector2d State_x; // 状态
	Eigen::Matrix2d covP_y;
	Eigen::Vector2d State_y;
	vector<double> memory_bias;

	System() : dqn(), mlp0(), mlp1(), mlp2(), mlp(), lstm(), last_action(0), count(0), covP_x(Eigen::Matrix2d::Identity()), State_x(Eigen::Vector2d::Zero()), covP_y(Eigen::Matrix2d::Identity()), State_y(Eigen::Vector2d::Zero()) {}
	int reset(vector<double> state) {
		covP_x << 1, 0, 0, 1; //协方差一开始为1，后续会迭代收敛
		State_x << state[0], 0; // 初始状态值， 第一个为位移、第二个为速度
		covP_y << 1, 0, 0, 1; //协方差一开始为1，后续会迭代收敛
		State_y << state[1], 0; // 初始状态值， 第一个为位移、第二个为速度
		memory_bias = { 0,0 };
		lstm.reset(state[0],state[1]);
		lstm1.reset(state[0], state[1]);
		lstm2.reset(state[0], state[1]);
		count = 0;
		vector<double> q_values = dqn.forward(state);
		last_action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
		return last_action;
	}

	int act(vector<double> state) {
		vector<double> STATE0 = lstm.forward({ state[0],state[1],v_x[last_action], v_y[last_action] });
		vector<double> STATE1 = lstm1.forward({ state[0],state[1],v_x[last_action], v_y[last_action] });
		vector<double> STATE2 = lstm2.forward({ state[0],state[1],v_x[last_action], v_y[last_action] });
		kf(state[0], v_x[last_action], covP_x, State_x);
		kf(state[1], v_y[last_action], covP_y, State_y);
		vector<double> STATE = { (STATE0[0] + STATE1[0] + STATE2[0] + 0 * State_x[0]) / 3, (STATE0[1] + STATE1[1] + STATE2[1] + 0 * State_y[0]) / 3};
		memory_bias.push_back(state[0] - STATE[0]);
		memory_bias.push_back(state[1] - STATE[1]);
		if (memory_bias.size() > 10) {
			memory_bias.erase(memory_bias.begin());
			memory_bias.erase(memory_bias.begin());
			if (mlp0.forward(memory_bias)[0] > 0.5 || mlp1.forward(memory_bias)[0] > 0.5 || mlp2.forward(memory_bias)[0] > 0.5 || mlp.forward(memory_bias)[0] > 0.5) {
				count++;
			}
			else {
				count = 0;
			}
			if (count >= 3) {
				return -1;
			}
		}
		state[0] = STATE[0];
		state[1] = STATE[1];
		State_x[0] = STATE[0];
		State_y[0] = STATE[1];
		vector<double> q_values = dqn.forward(state);
		last_action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
		return last_action;
	}

	vector<double> get_state() {
		return { State_x[0], State_y[0] };
	}

	void load() {
		dqn.load("trained models/dqn ");
		mlp.load("trained models/mlp ");
		mlp0.load("trained models/mlp0 ");
		mlp1.load("trained models/mlp1 ");
		mlp2.load("trained models/mlp2 ");
		lstm.load("trained models/lstm ");
		lstm1.load("trained models/lstm1 ");
		lstm2.load("trained models/lstm2 ");
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

void lstm_mlp_dqn_visual() {
	Env env;
	System system;
	system.load();
	initgraph(300, 300);
	env.reset(10);
	env.print_bkg();
	vector<double> state = env.observe();
	int action = system.reset(state);
	env.step(action);
	for (int step = 0; step < 1000; step++) {
		vector<double> state = env.observe();
		int action = system.act(state);
		if (action == -1) {
			Sleep(50);
			break;
		}
		env.step(action);
		env.print_obs_route();
		env.print_drone_route(50);
		if (env.is_done()) {
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
	double __bias[1000]{};
	int __count[1000]{};
	for (int episode = 0; episode < 10000; episode++) {
		env.reset(10);
		for (int step = 0; step < 1000; step++) {
			vector<double> state = env.observe();
			vector<double> q_values = dqn.forward(state);
			int action = max_element(q_values.begin(), q_values.end()) - q_values.begin();
			if (__filter_flag&&!env.is_break_down()) {
				__bias[step] += (state[0] - env.get_pos_x()) * (state[0] - env.get_pos_x()) + (state[1] - env.get_pos_y()) * (state[1] - env.get_pos_y());
				__count[step] += 2;
			}
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
	if (__filter_flag)
	{
		ofstream file("example.csv");
		for (int i = 0; i < 1000; i++) {
			if (__count[i] != 0)
				file << __bias[i] / __count[i] << ",";
		}
		file.close();
	}

}

void hidden_markov() {
	cout << "hidden markov model:" << endl;
	Env env;
	HM hm;
	hm.load();
	double avg_pos = 0;
	int success = 0;
	int break_count = 0;
	int success_break = 0;
	int good_stop = 0;
	int bad_stop = 0;
	int crash = 0;
	double __bias[1000]{};
	int __count[1000]{};
	for (int episode = 0; episode < 10000; episode++) {
		env.reset(10);
		vector<double> state = env.observe();
		int action = hm.reset(state);
		env.step(action);
		for (int step = 0; step < 1000; step++) {
			vector<double> state = env.observe();
			int action = hm.act(state);
			if (__filter_flag && !env.is_break_down()) {
				state = hm.get_state();
				__bias[step] += (state[0] - env.get_pos_x()) * (state[0] - env.get_pos_x()) + (state[1] - env.get_pos_y()) * (state[1] - env.get_pos_y());
				__count[step] += 2;
			}
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
	if (__filter_flag)
	{
		ofstream file("example.csv");
		for (int i = 0; i < 1000; i++) {
			if (__count[i] != 0)
				file << __bias[i] / __count[i] << ",";
		}
		file.close();
	}
}

void lstm_mlp_dqn() {
	cout << "lstm_mlp_dqn:" << endl;
	Env env;
	System system;
	system.load();
	int success = 0;
	int break_count = 0;
	int success_break = 0;
	int good_stop = 0;
	int bad_stop = 0;
	int crash = 0;
	double __bias[1000]{};
	int __count[1000]{};
	for (int episode = 0; episode < 10000; episode++) {
		env.reset(10);
		vector<double> state = env.observe();
		int action = system.reset(state);
		env.step(action);
		for (int step = 0; step < 1000; step++) {
			vector<double> state = env.observe();
			int action = system.act(state);
			if (__filter_flag && !env.is_break_down()) {
				state = system.get_state();
				__bias[step] += (state[0] - env.get_pos_x()) * (state[0] - env.get_pos_x()) + (state[1] - env.get_pos_y()) * (state[1] - env.get_pos_y());
				__count[step] += 2;
			}
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
	if(__filter_flag)
	{
		ofstream file("example.csv");
		for (int i = 0; i < 1000; i++) {
			if (__count[i] != 0)
				file << __bias[i] / __count[i] << ",";
		}
		file.close();
	}
}

int main() {
	while(1)
	{
		cout << "** 0.pure dqn (visual)    1.robust system (visual)    2.dqn only   3.robust system    4.hidden markov model  **" << endl;
		cout << "please input the number of the function you want to run:";
		int op;
		cin >> op;
		switch (op) {
			case 0:
				pure_dqn_visual();
				break;
			case 1:
				lstm_mlp_dqn_visual();
				break;
			case 2:
				pure_dqn();
				break;
			case 3:
				lstm_mlp_dqn();
				break;
			case 4:
				hidden_markov();
				break;
			default:
				return 0;
		}
		cout << endl;
		cout << "***************************************************************************************************************" << endl;
	}
	return 0;
}