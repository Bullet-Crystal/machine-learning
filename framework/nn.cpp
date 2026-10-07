#include "./imgui/imgui.h"
#include "./imgui/backends/imgui_impl_sdl3.h"
#include "./imgui/backends/imgui_impl_opengl3.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

#define USE_SOFTMAX 1

double rand_double(void){
	static std::mt19937 gen(std::random_device{}());
	static std::uniform_real_distribution<double> topo(0.0, 1.0);
	return topo(gen);
}

class Matrix{
	public:
		int num_rows;
		int num_cols;
		std::vector<double> data;

		Matrix(){
			this->num_cols = 0;
			this->num_rows = 0;
		}
		Matrix(int num_rows, int num_cols) {
			this->num_cols = num_cols;
			this->num_rows = num_rows;
			this->data = std::vector<double>((size_t) num_rows * num_cols, 0);
		}

		double& at(int i, int j) {
			assert(i >= 0 && i < num_rows && j >= 0 && j < num_cols);
			return data[(size_t) i * num_cols + j];
		}
		const double& at(int i, int j) const {
			assert(i >= 0 && i < num_rows && j >= 0 && j < num_cols);
			return data[(size_t) i * num_cols + j];
		}

		void init_random(double min, double max) {
			for (int i = 0; i < num_rows; ++i)
				for (int j = 0; j < num_cols; ++j)
					at(i, j) = rand_double() * (max - min) + min;
		}
};

double sigmoidd(double x) {
	return 1/(1 + std::exp(-x));
}

void softmax(Matrix& m) {
    for (int i = 0; i < m.num_rows; ++i) {
        double mx = m.at(i, 0);
        for (int j = 1; j < m.num_cols; ++j) mx = std::max(mx, m.at(i, j));
        double s = 0;
        for (int j = 0; j < m.num_cols; ++j) { m.at(i, j) = std::exp(m.at(i, j) - mx); s += m.at(i, j); }
        for (int j = 0; j < m.num_cols; ++j) m.at(i, j) /= s;
    }
}

Matrix mat_dot(const Matrix& a, const Matrix& b){
	assert(a.num_cols == b.num_rows);
	Matrix mat{a.num_rows, b.num_cols};
	for (int i = 0; i < a.num_rows; ++i)
		for (int k = 0; k < a.num_cols; ++k) {
			double mik = a.at(i, k);
			for (int j = 0; j < b.num_cols; ++j)
				mat.at(i, j) += mik * b.at(k, j);
		}
	return mat;
}

Matrix mat_sum(const Matrix& a, const Matrix& b) {
	assert(a.num_cols == b.num_cols && (a.num_rows == b.num_rows || b.num_rows == 1));
	Matrix mat{a.num_rows, a.num_cols};
	for (int i = 0; i < a.num_rows; ++i) {
		int bi = b.num_rows == 1 ? 0 : i;
		for (int j = 0; j < a.num_cols; ++j)
			mat.at(i, j) = a.at(i, j) + b.at(bi, j);
	}
	return mat;
}

Matrix mat_sub(const Matrix& a, const Matrix& b) {
	assert(a.num_rows == b.num_rows && a.num_cols == b.num_cols);
	Matrix mat{a.num_rows, a.num_cols};
	for (int i = 0; i < a.num_rows; ++i)
		for (int j = 0; j < a.num_cols; ++j)
			mat.at(i, j) = a.at(i, j) - b.at(i, j);
	return mat;
}

void mat_activate(Matrix& mat) {
	for (double& v : mat.data)
		v = sigmoidd(v);
}

Matrix mat_get_row(const Matrix& mat, int p) {
	assert(p >= 0 && p < mat.num_rows);
	Matrix temp = Matrix(1, mat.num_cols);
	std::copy(mat.data.begin() + (size_t) p * mat.num_cols,
			mat.data.begin() + (size_t) (p + 1) * mat.num_cols,
			temp.data.begin());
	return temp;
}
void mat_print(const Matrix& mat, const char* str) {
	std::cout<<str << " : [\n";
	for (int i = 0; i < mat.num_rows; ++i) {
		for (int j = 0; j < mat.num_cols; ++j)
			std::cout<<"    "<<mat.at(i, j) ;
		std::cout<<"\n";
	}
	std::cout<<"]\n";
}

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
		std::vector<int> topo;
		std::vector<Layer> layers;
		std::vector<Matrix> activations;

		Architecture() {}
		Architecture(std::vector<int> topo) {
			this->topo = topo;
			for (int i = 1; i < (int) topo.size(); ++i) {
				this->layers.push_back(
						Layer(
							Matrix(topo.at(i - 1), topo.at(i)),
							Matrix(1, topo.at(i)))
						);
			}
			this->init_layers();
			for (int i = 1; i < (int) topo.size(); ++i)
				this->activations.push_back(Matrix(1, topo.at(i)));
		}

		void init_layers() {
			for (int i = 0; i < (int) layers.size(); ++i) {
				Matrix& w = layers.at(i).weights;
				double lim = std::sqrt(6.0 / (w.num_rows + w.num_cols));
				w.init_random(-lim, lim);
			}
		}

		void zero() {
			for (int k = 0; k < (int) layers.size(); ++k) {
				std::fill(layers.at(k).weights.data.begin(), layers.at(k).weights.data.end(), 0.0);
				std::fill(layers.at(k).biases.data.begin(), layers.at(k).biases.data.end(), 0.0);
			}
		}
		void print() {
			for (int i = 0; i < (int) layers.size(); ++i) {
				std::cout<<"\nLayer " << i + 1 << std::endl;
				mat_print(layers.at(i).weights, "w");
				mat_print(layers.at(i).biases, "b");
				std::cout<<"\n";
			}
		}

Matrix forward(const Matrix& input) {
    Matrix output = input;
    for (int k = 0; k < (int) layers.size(); ++k) {
        output = mat_dot(output, layers.at(k).weights);
        output = mat_sum(output, layers.at(k).biases);
#if USE_SOFTMAX
        if (k == (int)layers.size() - 1) softmax(output);
        else mat_activate(output);
#else
        mat_activate(output);
#endif
        this->activations.at(k) = output;
    }
    return output;
}

		double cost(const Matrix& train_in, const Matrix& train_out) {
			double cost = 0;
			for (int i = 0; i < (int) train_in.num_rows; ++i) {
				Matrix input = mat_get_row(train_in, i);
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
				for (int i = 0; i < layers.at(k).weights.num_rows; ++i)
					for (int j = 0; j < layers.at(k).weights.num_cols; ++j)
						this->layers.at(k).weights.at(i, j) -= learning_rate * gradients.layers.at(k).weights.at(i, j);
				for (int i = 0; i < layers.at(k).biases.num_rows; ++i)
					for (int j = 0; j < layers.at(k).biases.num_cols; ++j)
						this->layers.at(k).biases.at(i, j) -= learning_rate * gradients.layers.at(k).biases.at(i, j);
			}
		}

		void test(const Matrix& train_in, const Matrix& train_out) {
			std::cout<<"\n=====================[TEST]==========================\n";
			for (int i = 0; i < (int) train_in.num_rows; ++i) {
				Matrix input = mat_get_row(train_in, i);
				Matrix mat = forward(input);
				Matrix dataected = mat_get_row(train_out, i);
				mat_print(input, "input");
				mat_print(mat, "output");
				mat_print(dataected, "dataected");
				std::cout<<"===================\n";
			}
		}

double backprop(Architecture& gradient, const Matrix& X, const Matrix& Y) {
    const int n = X.num_rows, L = (int)layers.size();
    gradient.zero();
    double total = 0;

    for (int i = 0; i < n; ++i) {
        Matrix input = mat_get_row(X, i);
        Matrix out   = forward(input);

        std::vector<double> da(Y.num_cols);
        for (int j = 0; j < Y.num_cols; ++j) {
#if USE_SOFTMAX
            total += -Y.at(i, j) * std::log(out.data[j] + 1e-12);
            da[j] = out.data[j] - Y.at(i, j);
#else
            double d = out.data[j] - Y.at(i, j);
            total += d * d;
            da[j] = 2 * d;
#endif
        }

        for (int l = L - 1; l >= 0; --l) {
            const Matrix& prev = (l == 0) ? input : activations[l - 1];
            const int nin = prev.num_cols, nout = activations[l].num_cols;

            std::vector<double> delta(nout);
            for (int j = 0; j < nout; ++j) {
                double a = activations[l].data[j];
                delta[j] = (USE_SOFTMAX && l == L - 1) ? da[j] : da[j] * a * (1 - a);
                gradient.layers[l].biases.data[j] += delta[j];
            }

            std::vector<double> da_prev(l > 0 ? nin : 0, 0.0);
            for (int k = 0; k < nin; ++k) {
                const double pk = prev.data[k];
                if (l == 0 && pk == 0.0) continue;          // most MNIST pixels are 0
                double* gw = &gradient.layers[l].weights.data[(size_t)k * nout];
                for (int j = 0; j < nout; ++j) gw[j] += delta[j] * pk;
                if (l > 0) {
                    const double* w = &layers[l].weights.data[(size_t)k * nout];
                    double acc = 0;
                    for (int j = 0; j < nout; ++j) acc += delta[j] * w[j];
                    da_prev[k] = acc;
                }
            }
            da = std::move(da_prev);
        }
    }
    for (int l = 0; l < L; ++l) {
        for (auto& v : gradient.layers[l].weights.data) v /= n;
        for (auto& v : gradient.layers[l].biases.data)  v /= n;
    }
    return total / n;
}
};


static uint32_t read_be32(std::ifstream& f) {
    unsigned char b[4]; f.read((char*)b, 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
}

Matrix load_images(const std::string& path, int limit = -1) {   // N x 784, values in [0,1]
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    if (read_be32(f) != 2051) throw std::runtime_error("bad image file");
    int n = read_be32(f), rows = read_be32(f), cols = read_be32(f);
    if (limit > 0 && limit < n) n = limit;
    Matrix m(n, rows * cols);
    std::vector<unsigned char> buf((size_t)rows * cols);
    for (int i = 0; i < n; ++i) {
        f.read((char*)buf.data(), buf.size());
        for (size_t j = 0; j < buf.size(); ++j) m.at(i, (int)j) = buf[j] / 255.0;
    }
    return m;
}

Matrix load_labels(const std::string& path, int limit = -1) {   // N x 10, one-hot
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    if (read_be32(f) != 2049) throw std::runtime_error("bad label file");
    int n = read_be32(f);
    if (limit > 0 && limit < n) n = limit;
    Matrix m(n, 10);
    for (int i = 0; i < n; ++i) { unsigned char c; f.read((char*)&c, 1); m.at(i, c) = 1.0; }
    return m;
}

int argmax(const Matrix& m, int row) {
    int best = 0;
    for (int j = 1; j < m.num_cols; ++j) if (m.at(row, j) > m.at(row, best)) best = j;
    return best;
}

void make_batch(const Matrix& X, const Matrix& Y, const std::vector<int>& order,
                int start, int B, Matrix& bx, Matrix& by) {
    for (int b = 0; b < B; ++b) {
        int r = order[start + b];
        std::copy(X.data.begin() + (size_t)r * X.num_cols, X.data.begin() + (size_t)(r + 1) * X.num_cols,
                  bx.data.begin() + (size_t)b * X.num_cols);
        std::copy(Y.data.begin() + (size_t)r * Y.num_cols, Y.data.begin() + (size_t)(r + 1) * Y.num_cols,
                  by.data.begin() + (size_t)b * Y.num_cols);
    }
}

double accuracy(Architecture& a, const Matrix& X, const Matrix& Y, int count) {
    int ok = 0;
    for (int i = 0; i < count; ++i)
			if (argmax(a.forward(mat_get_row(X, i)), 0) == argmax(Y, i)) ++ok;
    return ok / (double)count;
}

int main() {

	Matrix train_in  = load_images("./../mnist/train-images-idx3-ubyte");
	Matrix train_out = load_labels("./../mnist/train-labels-idx1-ubyte");
	Matrix test_in   = load_images("./../mnist/t10k-images-idx3-ubyte");
	Matrix test_out  = load_labels("./../mnist/t10k-labels-idx1-ubyte");

	const std::vector<int> topology{28 * 28, 64, 10};
	Architecture arch{topology};
	Architecture gradients = Architecture{arch};

	const int B = 32;
	Matrix bx(B, 784), by(B, 10);
	std::vector<int> order(train_in.num_rows);
	std::iota(order.begin(), order.end(), 0);
	std::mt19937 rng(42);
	std::shuffle(order.begin(), order.end(), rng);
	int pos = 0, step = 0;
	float test_acc = 0;

	
	if (!SDL_Init(SDL_INIT_VIDEO)) { printf("SDL error: %s\n", SDL_GetError()); return 1; }

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	SDL_Window* window = SDL_CreateWindow("XOR training", 1280, 720,
			SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
	if (!window) { printf("Window error: %s\n", SDL_GetError()); return 1; }

	SDL_GLContext gl = SDL_GL_CreateContext(window);
	SDL_GL_MakeCurrent(window, gl);
	SDL_GL_SetSwapInterval(1);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui::GetStyle().FontScaleMain = 1.5f;
	ImGui_ImplSDL3_InitForOpenGL(window, gl);
	ImGui_ImplOpenGL3_Init("#version 330");

	
	float lr = 1.0f;
	int steps_per_frame = 10;
	int view_points = 1000;
	bool training = false;
	std::vector<float> loss;
	bool running = true;
	float ema = -1.0f;

	while (running) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			ImGui_ImplSDL3_ProcessEvent(&e);
			if (e.type == SDL_EVENT_QUIT) running = false;
			if (e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
					e.window.windowID == SDL_GetWindowID(window)) running = false;
		}

		if (training) {
			for (int s = 0; s < steps_per_frame; ++s) {
				if (pos + B > (int)order.size()) { std::shuffle(order.begin(), order.end(), rng); pos = 0; }
				make_batch(train_in, train_out, order, pos, B, bx, by);
				pos += B;
				arch.backprop(gradients, bx, by);
				float l = (float)arch.cost(bx, by);
				ema = (ema < 0.0f) ? l : 0.98f * ema + 0.02f * l;
				loss.push_back(ema);
				arch.learn(gradients, lr);
				if (++step % 100 == 0) test_acc = (float)accuracy(arch, test_in, test_out, 1000);
			}
		}

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Controls");
		ImGui::SliderFloat("Learning rate", &lr, 0.01f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		ImGui::SliderInt("Steps / frame", &steps_per_frame, 1, 500);
		if (ImGui::Button(training ? "Pause" : "Resume")) training = !training;
		ImGui::SameLine();
		if (ImGui::Button("Reset")) {
			arch = Architecture{topology};
			gradients = Architecture{arch};
			loss.clear();
		}
		ImGui::SameLine();
		if (ImGui::Button("Print test (stdout)")) {
			arch.print();
			arch.test(train_in, train_out);
		}

		int n = (int) loss.size();
		ImGui::PlotLines("##loss", loss.data(), n, 0,
				"Loss", 0.0f, 0.3f, ImVec2(-1, 220));
		ImGui::Text("epoch %d   loss %.6f", n, n ? loss.back() : 0.0f);
		ImGui::Text("test accuracy (1000 samples): %.1f%%", test_acc * 100);

		ImGui::Separator();
		// Display Mnist Test

		static int sample = 0;
		const int last = test_in.num_rows - 1;

		if (!ImGui::GetIO().WantTextInput) {              // ignore keys while typing in a text field
			int step = ImGui::GetIO().KeyShift ? 10 : 1;  // hold Shift to jump 10 samples
			if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) sample = std::min(sample + step, last);
			if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))  sample = std::max(sample - step, 0);
		}

		ImGui::Text("Test sample: %d / %d   (left/right arrows, Shift = 10)", sample, last);
		Matrix out = arch.forward(mat_get_row(test_in, sample));

		ImVec2 p = ImGui::GetCursorScreenPos();
		const float cell = 10.0f;
		ImDrawList* dl = ImGui::GetWindowDrawList();
		for (int y = 0; y < 28; ++y)
			for (int x = 0; x < 28; ++x) {
				int v = (int)(test_in.at(sample, y * 28 + x) * 255);
				dl->AddRectFilled(ImVec2(p.x + x * cell, p.y + y * cell),
						ImVec2(p.x + (x + 1) * cell, p.y + (y + 1) * cell),
						IM_COL32(v, v, v, 255));
			}
		ImGui::Dummy(ImVec2(28 * cell, 28 * cell));

		float probs[10];
		for (int j = 0; j < 10; ++j) probs[j] = (float)out.at(0, j);
		ImGui::PlotHistogram("##out", probs, 10, 0, "outputs 0-9", 0.0f, 1.0f, ImVec2(280, 100));
		ImGui::Text("predicted %d, label %d", argmax(out, 0), argmax(test_out, sample));
		ImGui::End();

		ImGui::Render();
		int w, h;
		SDL_GetWindowSizeInPixels(window, &w, &h);
		glViewport(0, 0, w, h);
		glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		SDL_GL_SwapWindow(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();
	SDL_GL_DestroyContext(gl);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
