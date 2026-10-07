#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

double rand_double(void){

	static std::mt19937 gen(std::random_device{}());
	static std::uniform_real_distribution<double> dist(0.0, 1.0);
	return dist(gen);
}

class Matrix{
	public:
		int num_rows;
		int num_cols;

		std::vector<double> exp;

		Matrix(){
			this->num_cols = 0;
			this->num_rows = 0;
		}
		Matrix(int num_rows, int num_cols) {
			this->num_cols = num_cols;
			this->num_rows = num_rows;
			this->exp = std::vector<double>((size_t) num_rows * num_cols, 0);
		}


		double& at(int i, int j) {
			assert(i >= 0 && i < num_rows && j >= 0 && j < num_cols);
			return exp[(size_t) i * num_cols + j];
		}
		const double& at(int i, int j) const {
			assert(i >= 0 && i < num_rows && j >= 0 && j < num_cols);
			return exp[(size_t) i * num_cols + j];
		}

		void initRandom(double min, double max) {
			for (int i = 0; i < num_rows; ++i) {
				for (int j = 0; j < num_cols; ++j) {
					at(i, j) = rand_double() * (max - min) + min;
				}
			}
		}
};

double sigmoidd(double x) {
	return 1/(1 + std::exp(-x));
}

class ServiceMatrix {
	public:
		ServiceMatrix(){}

		Matrix dot(const Matrix& a, const Matrix& b){
			assert(a.num_cols == b.num_rows);
			Matrix mat{a.num_rows, b.num_cols};

			for (int i = 0; i < a.num_rows; ++i) {
				for (int k = 0; k < a.num_cols; ++k) {
					double mik = a.at(i, k);
					for (int j = 0; j < b.num_cols; ++j) {
						mat.at(i, j) += mik * b.at(k, j);
					}
				}
			}
			return mat;
		}


		Matrix sum(const Matrix& a, const Matrix& b) {
			assert(a.num_cols == b.num_cols && (a.num_rows == b.num_rows || b.num_rows == 1));
			Matrix mat{a.num_rows, a.num_cols};

			for (int i = 0; i < a.num_rows; ++i) {
				int bi = b.num_rows == 1 ? 0 : i;
				for (int j = 0; j < a.num_cols; ++j) {
					mat.at(i, j) = a.at(i, j) + b.at(bi, j);
				}
			}
			return mat;
		}

		Matrix sub(const Matrix& a, const Matrix& b) {
			assert(a.num_rows == b.num_rows && a.num_cols == b.num_cols);
			Matrix mat{a.num_rows, a.num_cols};

			for (int i = 0; i < a.num_rows; ++i) {
				for (int j = 0; j < a.num_cols; ++j) {
					mat.at(i, j) = a.at(i, j) - b.at(i, j);
				}
			}
			return mat;
		}


		void activate(Matrix& mat) {
			for (double& v : mat.exp)
				v = sigmoidd(v);
		}

		Matrix get_row(const Matrix& mat, int p) {
			assert(p >= 0 && p < mat.num_rows);
			Matrix temp = Matrix(1, mat.num_cols);
			std::copy(mat.exp.begin() + (size_t) p * mat.num_cols,
					mat.exp.begin() + (size_t) (p + 1) * mat.num_cols,
					temp.exp.begin());
			return temp;
		}
		void print(const Matrix& mat, const char* str) {
			std::cout<<str << " : [\n";
			for (int i = 0; i < mat.num_rows; ++i) {
				for (int j = 0; j < mat.num_cols; ++j) {
					std::cout<<"    "<<mat.at(i, j) ;
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
		std::vector<Matrix> activations;

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
			this->init_layers();
			for (int i = 1; i < (int) dist.size(); ++i) {
				this->activations.push_back(Matrix(1, dist.at(i)));
			}
		}

		void init_layers() {
			for (int i = 0; i < (int) layers.size(); ++i) {

				Matrix& w = layers.at(i).weights;
				double lim = std::sqrt(6.0 / (w.num_rows + w.num_cols));
				w.initRandom(-lim, lim);
			}
		}

		void zero() {
			for (int k = 0; k < (int) layers.size(); ++k) {
				std::fill(layers.at(k).weights.exp.begin(), layers.at(k).weights.exp.end(), 0.0);
				std::fill(layers.at(k).biases.exp.begin(), layers.at(k).biases.exp.end(), 0.0);
			}
		}
		void print() {
			for (int i = 0; i < (int) layers.size(); ++i) {
				std::cout<<"\nLayer " << i + 1 << std::endl;
				serviceMat.print(layers.at(i).weights, "w");
				serviceMat.print(layers.at(i).biases, "b");
				std::cout<<"\n";
			}
		}
		Matrix forward(const Matrix& input) {
			Matrix output = input;
			for (int k = 0; k < (int) layers.size(); ++k) {
				output = serviceMat.dot(output, layers.at(k).weights);
				output = serviceMat.sum(output, layers.at(k).biases);
				serviceMat.activate(output);
				this->activations.at(k) = output;
			}
			return output;
		}

		double cost(const Matrix& train_in, const Matrix& train_out) {
			double cost = 0;
			for (int i = 0; i < (int) train_in.num_rows; ++i) {
				Matrix input = serviceMat.get_row(train_in, i);
				Matrix f = forward(input);
				for (int j = 0; j < train_out.num_cols; ++j) {
					double d = (f.at(0, j) - train_out.at(i, j));
					cost += d*d;
				}
			}
			return cost / train_in.num_rows;
		}
		void learn(const Architecture& gradients, double learning_rate = 1) {
			for (int k = 0; k < (int) layers.size(); ++k) {
				for (int i = 0; i < layers.at(k).weights.num_rows; ++i) {
					for (int j = 0; j < layers.at(k).weights.num_cols; ++j) {
						this->layers.at(k).weights.at(i, j) -= learning_rate * gradients.layers.at(k).weights.at(i, j);
					}
				}
				for (int i = 0; i < layers.at(k).biases.num_rows; ++i) {
					for (int j = 0; j < layers.at(k).biases.num_cols; ++j) {
						this->layers.at(k).biases.at(i, j) -= learning_rate * gradients.layers.at(k).biases.at(i, j);
					}
				}
			}
		}

		void test(const Matrix& train_in, const Matrix& train_out) {
			std::cout<<"\n=====================[TEST]==========================\n";
			for (int i = 0; i < (int) train_in.num_rows; ++i) {
				Matrix input = serviceMat.get_row(train_in, i);
				Matrix mat = forward(input);
				Matrix expected = serviceMat.get_row(train_out, i);
				serviceMat.print(input, "input");
				serviceMat.print(mat, "output");
				serviceMat.print(expected, "expected");
				std::cout<<"===================\n";
			}
		}

		void finite_diff(Architecture& gradients, const Matrix& train_in, const Matrix& train_out, double eps = 1e-5) {
			for (int k = 0; k < (int) layers.size(); ++k) {
				for (int i = 0; i < layers.at(k).weights.num_rows; ++i) {
					for (int j = 0; j < layers.at(k).weights.num_cols; ++j) {
						double saved = layers.at(k).weights.at(i, j);
						layers.at(k).weights.at(i, j) = saved + eps;
						double c_plus = cost(train_in, train_out);
						layers.at(k).weights.at(i, j) = saved - eps;
						double c_minus = cost(train_in, train_out);
						layers.at(k).weights.at(i, j) = saved;
						gradients.layers.at(k).weights.at(i, j) = (c_plus - c_minus) / (2 * eps);
					}
				}
			}
			for (int k = 0; k < (int) layers.size(); ++k) {
				for (int i = 0; i < layers.at(k).biases.num_rows; ++i) {
					for (int j = 0; j < layers.at(k).biases.num_cols; ++j) {
						double saved = layers.at(k).biases.at(i, j);
						layers.at(k).biases.at(i, j) = saved + eps;
						double c_plus = cost(train_in, train_out);
						layers.at(k).biases.at(i, j) = saved - eps;
						double c_minus = cost(train_in, train_out);
						layers.at(k).biases.at(i, j) = saved;
						gradients.layers.at(k).biases.at(i, j) = (c_plus - c_minus) / (2 * eps);
					}
				}
			}
		}

		void backprop(Architecture& gradient, const Matrix& X, const Matrix& Y) {
			const int n = X.num_rows, L = (int)layers.size();
			gradient.zero();

			for (int i = 0; i < n; ++i) {
				Matrix input = serviceMat.get_row(X, i);
				Matrix out   = forward(input);

				std::vector<double> da(Y.num_cols);
				for (int j = 0; j < Y.num_cols; ++j)
					da[j] = 2 * (out.at(0, j) - Y.at(i, j)); 

				for (int l = L - 1; l >= 0; --l) {
					const Matrix& prev = (l == 0) ? input : activations[l-1];
					std::vector<double> da_prev(prev.num_cols, 0.0);
					for (int j = 0; j < activations[l].num_cols; ++j) {
						double a = activations[l].at(0, j);
						double delta = da[j] * a * (1 - a);
						gradient.layers[l].biases.at(0, j) += delta;
						for (int k = 0; k < prev.num_cols; ++k) {
							gradient.layers[l].weights.at(k, j) += delta * prev.at(0, k);
							if (l > 0)
								da_prev[k] += delta * layers[l].weights.at(k, j);
						}
					}
					da = std::move(da_prev);
				}
			}
			for (int l = 0; l < L; ++l) {
				for (auto& v : gradient.layers[l].weights.exp) v /= n;
				for (auto& v : gradient.layers[l].biases.exp)  v /= n;
			}
		}
};

int main() {

	ServiceMatrix serviceMat{};

	Matrix train_in = Matrix(4, 2);
	Matrix train_out = Matrix(4, 1);
	train_in.exp = {
		0, 0,
		0, 1,
		1, 0,
		1, 1
	};

	train_out.exp = {
		0,
		1,
		1,
		0
	};
	Architecture arch{std::vector<int>{2, 100, 1}};
	arch.print();
	Architecture gradients = Architecture{arch};

	double learning_rate = 1;
	int MAX_STEP = 100*1000;
	for (int epoch = 0; epoch <= MAX_STEP; ++epoch) {

		arch.backprop(gradients, train_in, train_out);
		arch.learn(gradients, learning_rate);

		double progress = (epoch / (double) MAX_STEP ) * 100;

		std::cout<< "\r" << "cost ( " << progress << "% ) = " << arch.cost(train_in, train_out) << std::flush;
	}
	arch.print();
	arch.test(train_in, train_out);
	return 0;
}
