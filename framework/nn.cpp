#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <vector>

double rand_double(void){
	// return a double between 0 and 1
	return (double) rand()/(double) RAND_MAX;
}

class Matrix{
	public:
		int num_rows;
		int num_cols;
		std::vector<std::vector<double>> exp;

		Matrix(){}
		Matrix(int num_rows, int num_cols) {
			this->num_cols = num_cols;
			this->num_rows = num_rows;
			this->exp = std::vector(num_rows, std::vector<double>(num_cols, 0));
		}

		void initRandom(double min, double max) {
			for (int i = 0; i < num_rows; ++i) {
				for (int j = 0; j < num_cols; ++j) {
					exp.at(i).at(j) = rand_double() * (max - min) + min;
				}
			}
		}
};

double sigmoidd(double x) {
	return 1/(1 + std::exp(-x));
}

double relu(double x) {
	return x > 0 ? x : 0;
}
class ServiceMatrix {
	public:
		ServiceMatrix(){}

		Matrix dot(Matrix& a, Matrix& b){
			Matrix mat{a.num_rows, b.num_cols};
			assert(a.num_cols == b.num_rows);

			for (int i = 0; i < a.num_rows; ++i) {
				for (int k = 0; k < a.num_cols; ++k) {
					double mik = a.exp.at(i).at(k);
					for (int j = 0; j < b.num_cols; ++j) {
						mat.exp.at(i).at(j) += mik * b.exp.at(k).at(j);
					}
				}
			}
			return mat;
		}

		Matrix sum(Matrix& a, Matrix& b) {
			assert(a.num_rows == b.num_rows && a.num_cols == b.num_cols);
			Matrix mat{a.num_rows, b.num_cols};

			for (int i = 0; i < a.num_rows; ++i) {
				for (int j = 0; j < b.num_cols; ++j) {
					mat.exp.at(i).at(j) = a.exp.at(i).at(j) + b.exp.at(i).at(j);
				}
			}
			return mat;
		}

		Matrix sub(Matrix& a, Matrix& b) {
			assert(a.num_rows == b.num_rows && a.num_cols == b.num_cols);
			Matrix mat{a.num_rows, b.num_cols};

			for (int i = 0; i < a.num_rows; ++i) {
				for (int j = 0; j < b.num_cols; ++j) {
					mat.exp.at(i).at(j) = a.exp.at(i).at(j) - b.exp.at(i).at(j);
				}
			}
			return mat;
		}
		Matrix activate(Matrix& mat) {
			for (int i = 0; i < mat.num_rows; ++i) {
				for (int j = 0; j < mat.num_cols; ++j) {
					mat.exp.at(i).at(j) = relu(mat.exp.at(i).at(j));
				}
			}
			return mat;
		}

		Matrix get_row(Matrix& mat, int p) {
			Matrix temp = mat;
			temp.num_rows = 1;
			temp.exp.at(0) = mat.exp.at(p);
			return temp;
		}
		Matrix get_col(Matrix& mat, int p) {
			Matrix temp = mat;
			temp.num_cols = 1;
			for (int i = 0; i < mat.num_cols; ++i)
				temp.exp.at(i).at(0) = mat.exp.at(i).at(p);
			return temp;
		}
		void print(Matrix& mat, const char* str) {
			std::cout<<str << " : [\n";
			for (int i = 0; i < mat.num_rows; ++i) {
				for (int j = 0; j < mat.num_cols; ++j) {
					std::cout<<"    "<<mat.exp.at(i).at(j) ;
				}
				std::cout<<"\n";
			}
			std::cout<<"]\n";
		}
};

class Layer{
	public:
		Matrix weights;
		Matrix biases;

		Layer() {}

		Layer(Matrix weights, Matrix biases) {
			this->weights = weights;
			this->biases = biases;
		}
};
class Architecture{

	public:
		ServiceMatrix serviceMat{};

		std::vector<int> dist;
		std::vector<Layer> layers;

		Architecture() {
		}
		Architecture(std::vector<int> dist) {
			this->dist = dist;

			for (int i = 1; i < (int) dist.size(); ++i) {
				this->layers.push_back(
						Layer(
							Matrix(dist.at(i - 1), dist.at(i)),
							Matrix(1, dist.at(i)))
						);
			}
			this->initLayers();
		}
		void initLayers() {
			for (int i = 0; i < (int) layers.size(); ++i) {
				layers.at(i).weights.initRandom(0, 1);
				layers.at(i).biases.initRandom(0, 1);
			}
		}
		void print() {
			for (int i = 0; i < (int) layers.size(); ++i) {
				std::cout<<"Layer " << i+1 << std::endl;
				serviceMat.print(layers.at(i).weights, "w");
				serviceMat.print(layers.at(i).biases, "b");
				std::cout<<"\n";
			}
		}
		Matrix forward(Matrix& input) {
				Matrix output = input;
				for (int k = 0; k < (int) layers.size(); ++k) {
					output = serviceMat.dot(output, layers.at(k).weights);
					output = serviceMat.sum(output, layers.at(k).biases);
					output = serviceMat.activate(output);
				}
				return output;
		}
		double cost(Matrix& train_in, Matrix& train_out) {
			double cost = 0;
			for (int i = 0; i < (int) train_in.num_rows; ++i) {
				double cost_vect = 0;
				Matrix input = serviceMat.get_row(train_in, i);
				Matrix f = forward(input);
				for (int j = 0; j < train_out.num_cols; ++j) {
					double d = (f.exp.at(0).at(j) - train_out.exp.at(i).at(j));
					cost_vect += d*d;
				}
				cost += cost_vect / train_out.num_cols;
			}
			return cost / train_in.num_rows;
		}
		void learn(std::vector<Layer> gradiants, double learning_rate = 0.001) {
				for (int k = 0; k < (int) layers.size(); ++k) {
					for (int i = 0; i < layers.at(k).weights.num_rows; ++i) {
						for (int j = 0; j < layers.at(k).weights.num_cols; ++j) {
							this->layers.at(k).weights.exp.at(i).at(j) -= learning_rate * gradiants.at(k).weights.exp.at(i).at(j);
						}
					}
					for (int i = 0; i < layers.at(k).biases.num_rows; ++i) {
						for (int j = 0; j < layers.at(k).biases.num_cols; ++j) {
							this->layers.at(k).biases.exp.at(i).at(j) -= learning_rate * gradiants.at(k).biases.exp.at(i).at(j);
						}
					}
				}
		}

		void test(Matrix& train_in, Matrix& train_out) {
			std::cout<<"===============================================\n";
			print();
			for (int i = 0; i < (int) train_in.num_rows; ++i) {
				Matrix input = serviceMat.get_row(train_in, i);
				Matrix mat = forward(input);
				Matrix expected = serviceMat.get_row(train_out, i);
			std::cout<<"========================\n";
				serviceMat.print(input, "input");
			std::cout<<"========================\n";
				serviceMat.print(mat, "output");
			std::cout<<"========================\n";
				serviceMat.print(expected, "expected");
			std::cout<<"===============================================\n";
			}
		}
		void finite_diff(Matrix& train_in, Matrix& train_out, double eps = 0.0001) {
			std::vector<Layer> layers_gradiants;
			for (int i = 1; i < (int) dist.size(); ++i) {
				layers_gradiants.push_back(
						Layer(
							Matrix(dist.at(i - 1), dist.at(i)),
							Matrix(1, dist.at(i)))
						);
			}
			for (int epoch = 0; epoch < 10*1000; ++epoch) {
				double c = cost(train_in, train_out);
				for (int k = 0; k < (int) layers.size(); ++k) {
					for (int i = 0; i < layers.at(k).weights.num_rows; ++i) {
						for (int j = 0; j < layers.at(k).weights.num_cols; ++j) {
							layers.at(k).weights.exp.at(i).at(j) += eps;
							double c_new = cost(train_in, train_out);
							layers_gradiants.at(k).weights.exp.at(i).at(j) = (c_new - c) / eps;
							layers.at(k).weights.exp.at(i).at(j) -= eps;
						}
					}
				}
				for (int k = 0; k < (int) layers.size(); ++k) {
					for (int i = 0; i < layers.at(k).biases.num_rows; ++i) {
						for (int j = 0; j < layers.at(k).biases.num_cols; ++j) {
							layers.at(k).biases.exp.at(i).at(j) += eps;
							double c_new = cost(train_in, train_out);
							layers_gradiants.at(k).biases.exp.at(i).at(j) = (c_new - c) / eps;
							layers.at(k).biases.exp.at(i).at(j) -= eps;
						}
					}
				}
				learn(layers_gradiants);
				if (epoch % 100 == 0) {
					std::cout<< "\n" << "cost (epoch " << epoch << " ) = " << cost(train_in, train_out);
				}

			}
			test(train_in, train_out);

		}
};

int main() {

	ServiceMatrix serviceMat{};

	srand(time(0));

	std::vector<std::vector<int>> training_data = {
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 1}
	};

	Matrix train_in = Matrix(4, 2);
	Matrix train_out = Matrix(4, 1);
	train_in.exp = {
		{1, 3},
		{2, 4},
		{3, 5},
		{4, 6}
	};

	train_out.exp = {
		{4},
		{6},
		{8},
		{10}
	};
	Architecture arch{std::vector<int>{2, 2, 3, 2, 1}};
	arch.print();
	arch.finite_diff(train_in, train_out);
	return 0;
}
