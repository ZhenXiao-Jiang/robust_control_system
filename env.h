#pragma once
#include "J_random.h"
#include <vector>

#include <graphics.h>		// 引用图形库头文件
#include <conio.h>

class Env {
private:
	double _pos_x, _pos_y;
	int _pos;
	double _tar_x, _tar_y;
	int _step_count;
	bool _is_done;
	int _break_down, _break_mode;
	double _obs_x, _obs_y;
public:
	Env(double pos_x = 0, double pos_y = 0, double tar_x = 0, double tar_y = 0) {
		_pos_x = pos_x;
		_pos_y = pos_y;
		_obs_x = pos_x;
		_obs_y = pos_y;
		_pos = 0;
		_tar_x = tar_x;
		_tar_y = tar_y;
		_step_count = 0;
		_is_done = false;
		_break_down = -1;
		_break_mode = -1;
	}

	void print_bkg() {
		setbkcolor(WHITE);		// 设置背景色为白色
		cleardevice();			// 清屏
		//墙
		setlinecolor(BLACK);		// 设置画笔颜色为黑色
		setlinestyle(PS_SOLID, 2);	// 设置画笔线型为实线
		setfillcolor(GREEN);
		rectangle(110, 110, 190, 190);
		rectangle(70, 70, 230, 230);
		rectangle(30, 30, 270, 270);
		//门
		setlinecolor(WHITE);
		line(110, 180, 110, 190);
		line(120, 110, 140, 110);
		line(190, 120, 190, 160);
		line(150, 190, 180, 190);
		line(70, 170, 70, 180);
		line(70, 90, 70, 130);
		line(70, 210, 70, 230);
		line(230, 80, 230, 110);
		line(230, 200, 230, 220);
		line(100, 70, 110, 70);
		line(140, 70, 160, 70);
		line(190, 70, 230, 70);
		line(70, 230, 80, 230);
		line(120, 230, 150, 230);
		line(210, 230, 230, 230);
		//目标
		setlinecolor(GREEN);
		setfillcolor(GREEN);
		fillrectangle(_tar_x * 10 + 150 - 15 + 1, _tar_y * 10 + 150 - 15 + 1, _tar_x * 10 + 150 + 15 - 1, _tar_y * 10 + 150 + 15 - 1);
	}

	void print_drone_pixel(int sleep_time = 200) {
		putpixel(_pos_x * 10 + 150, _pos_y * 10 + 150, RED);
		Sleep(sleep_time);
		putpixel(_pos_x * 10 + 150, _pos_y * 10 + 150, WHITE);
	}

	void print_drone_circle(int sleep_time = 200) {
		setlinecolor(RED);
		setfillcolor(RED);
		fillcircle(_pos_x * 10 + 150, _pos_y * 10 + 150, 2);
		Sleep(sleep_time);
		setlinecolor(WHITE);
		setfillcolor(WHITE);
		fillcircle(_pos_x * 10 + 150, _pos_y * 10 + 150, 2);
	}

	void print_drone_route(int sleep_time = 200) {
		setlinecolor(RED);
		setfillcolor(RED);
		fillcircle(_pos_x * 10 + 150, _pos_y * 10 + 150, 1);
		Sleep(sleep_time);
	}

	void print_obs_route(int sleep_time = 0) {
		setlinecolor(BLUE);
		setfillcolor(BLUE);
		fillcircle(_obs_x * 10 + 150, _obs_y * 10 + 150, 1);
		Sleep(sleep_time);
	}

	void reset(int tar = -1, double pos_x = 0, double pos_y = 0, int pos = 0) {
		_pos_x = pos_x + int_random(-2, 2);
		_pos_y = pos_y + int_random(-2, 2);
		_obs_x = _pos_x;
		_obs_y = _pos_y;
		_pos = pos;
		if(tar == -1){
			tar = int_random(0, 19);
		}
		if (tar < 5) {
			_tar_x = -10;
			_tar_y = 10 - 4 * tar;
		}
		else if (tar < 10) {
			_tar_x = 10;
			_tar_y = - 10 + 4 * (tar - 5);
		}
		else if (tar < 15) {
			_tar_x = - 10 + 4 * (tar - 10);
			_tar_y = -10;
		}
		else {
			_tar_x = 10 - 4 * (tar - 15);
			_tar_y = 10;
		}
		_step_count = 0;
		_is_done = false;
		_break_down = int_random(10, 300);
		_break_mode = int_random(0, 2);
	}

	double get_pos_x() const {
		return _pos_x;
	}

	double get_pos_y() const {
		return _pos_y;
	}

	int get_pos() const {
		return _pos;
	}

	double get_tar_x() const {
		return _tar_x;
	}

	double get_tar_y() const {
		return _tar_y;
	}

	bool is_done() const {
		return _is_done;
	}

	bool is_break_down() const {
		if(_break_down == 0){
			return 1;
		}
		return 0;
	}

	double step(int action) {
		double dx = 0, dy = 0, vel = 0.3 + double_random_dis_0(0,1,2) * 0.05;
		if (action == 0) {
			dx = -vel;
		}
		else if (action == 1) {
			dx = vel;
		}
		else if (action == 2) {
			dy = vel;
		}
		else {
			dy = -vel;
		}
		_pos_x += dx;
		_pos_y += dy;

		//碰撞检测
		if (_pos == 0) {
			if (_pos_x < -4) {
				if (_pos_y < 3) {
					_is_done = true;
				}
				else {
					_pos = 1;
				}
			}
			else if (_pos_x > 4) {
				if (_pos_y < -3 || _pos_y > 1) {
					_is_done = true;
				}
				else {
					_pos = 1;
				}
			}
			else if (_pos_y > 4) {
				if (_pos_x > 3 || _pos_x < 0) {
					_is_done = true;
				}
				else {
					_pos = 1;
				}
			}
			else if (_pos_y < -4) {
				if (_pos_x < -3 || _pos_x > -1) {
					_is_done = true;
				}
				else {
					_pos = 1;
				}
			}
		}
		else if (_pos == 1) {
			if (abs(_pos_x) < 4 && abs(_pos_y) < 4) {
				_pos = 0;
			}
			else if (_pos_x < -8) {
				if ((_pos_y < -2 && _pos_y > -6) || (_pos_y < 3 && _pos_y > 2) || _pos_y > 6) {
					_pos = 2;
				}
				else {
					_is_done = true;
				}
			}
			else if (_pos_x > 8) {
				if ((_pos_y > 5 && _pos_y < 7) || (_pos_y > -7 && _pos_y < -4)) {
					_pos = 2;
				}
				else {
					_is_done = true;
				}
			}
			else if (_pos_y < -8) {
				if ((_pos_x > -5 && _pos_x < -4) || (_pos_x > -1 && _pos_x < 1) || _pos_x > 4) {

					_pos = 2;
				}
				else {
					_is_done = true;
				}
			}
			else if (_pos_y > 8) {
				if ((_pos_x > -3 && _pos_x < 0) || _pos_x < -7 || _pos_x > 6) {
					_pos = 2;
				}
				else {
					_is_done = true;
				}
			}
		}
		else if (_pos == 2) {
			if (abs(_pos_x) > 12 || abs(_pos_y) > 12) {
				_is_done = true;
			}
			else if (abs(_pos_x) < 8 && abs(_pos_y) < 8) {
				_pos = 1;
			}
			else if (abs(_pos_x - _tar_x) < 1.5 && abs(_pos_y - _tar_y) < 1.5) {
				_is_done = true;
				_pos = 3;
			}
		}

		_step_count++;
		if (_step_count > 100) {
			_is_done = true;
		}
		return 1;
	}

	std::vector<double> get_state() {
		std::vector<double> state = { _pos_x, _pos_y, _tar_x, _tar_y };
		return state;
	}

	std::vector<double> observe() {
		if (_break_down) {
			_break_down--;
			_obs_x = _pos_x + 0.8 * double_random_dis_0(0, 3, 5) + 0.2 * int_random(-10, 10) * 0.1;
			_obs_y = _pos_y + 0.8 * double_random_dis_0(0, 3, 5) + 0.2 * int_random(-10, 10) * 0.1;
			//_obs_x = _pos_x + 0.8 * double_random_dis_0(0, 2, 3) + 0.2 * int_random(-10, 10) * 0.1;
			//_obs_y = _pos_y + 0.8 * double_random_dis_0(0, 2, 3) + 0.2 * int_random(-10, 10) * 0.1;
			return {_obs_x, _obs_y, _tar_x, _tar_y};
		}
		else {
			switch (_break_mode) {
			case 0:
				return { int_random(-120, 120) / 10.0, int_random(-120, 120) / 10.0, _tar_x, _tar_y };
			case 1:
				return { _obs_x, _obs_y, _tar_x, _tar_y };
			case 2:
				_break_mode = int_random(6, 10);
			default:
				_obs_x = _pos_x + 0.8 * double_random_dis_0(0, 3, 5) + 0.2 * int_random(-10, 10) * 0.1;
				_obs_y = _pos_y + 0.8 * double_random_dis_0(0, 3, 5) + 0.2 * int_random(-10, 10) * 0.1;
				//_obs_x = _pos_x + 0.8 * double_random_dis_0(0, 2, 3) + 0.2 * int_random(-10, 10) * 0.1;
				//_obs_y = _pos_y + 0.8 * double_random_dis_0(0, 2, 3) + 0.2 * int_random(-10, 10) * 0.1;
				return { _obs_x + _break_mode, _obs_y, _tar_x, _tar_y };
			}
		}
		printf("observe error\n");
		exit(1);
	}

	std::vector<double> processed_observe() {
		std::vector<double> state = { _pos_x + double_random_dis_0(0,1,1), _pos_y + double_random_dis_0(0,1,1), _tar_x, _tar_y };
		return state;
	}
};