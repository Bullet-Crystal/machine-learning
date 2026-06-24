#include <cmath>
#include <cstdlib>
#include <iostream>
#include <time.h>


float train[][3] = {
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0}
};
#define TRAINING_SIZE (int)(sizeof(train)/sizeof(train[0]))
#define PARAMS_SIZE (int)2
#define INPUT_LAYER_SIZE (int)2
#define OUTPUT_LAYER_SIZE (int)1


float rand_float(void);

struct Neuron {
    float w[PARAMS_SIZE];
    float b;

	Neuron(){
		for(int i = 0; i < PARAMS_SIZE; ++i){
			this->w[i] = rand_float();
		}
		this->b = rand_float();
	}
};


struct Model {
	Neuron* input_layer;
	Neuron* output_layer;

	Model(){
		this->input_layer = new Neuron[INPUT_LAYER_SIZE];
		this->output_layer = new Neuron[OUTPUT_LAYER_SIZE];
	}

	~Model(){
		delete[] input_layer;
		delete[] output_layer;
	}
};
// generate a random float between 0 and 1
float rand_float(void){
	return (float) rand()/(float) RAND_MAX;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}
float compute(const Model& model){

	float result = 0.0f;
	for(int i = 0; i < TRAINING_SIZE; ++i){
		Neuron* il_neurons = model.input_layer;
		float a = il_neurons[0].b + il_neurons[0].w[0] * train[i][0] + il_neurons[0].w[1] * train[i][1];
		float b = il_neurons[1].b + il_neurons[1].w[0] * train[i][0] + il_neurons[1].w[1] * train[i][1];
		float actual = model.output_layer[0].b + model.output_layer[0].w[0] * sigmoid(a) + model.output_layer[0].w[1] * sigmoid(b);
		float expected = train[i][PARAMS_SIZE];
		float loss = (expected - sigmoid(actual));
		result += loss * loss;
	}
	result /= (float)TRAINING_SIZE;
	return result;
}

void derive(const Model& model, float direction = 0.0001f, float learning_rate = 0.1f){
	float computed = compute(model);
	float gradiants_i_w[2][2];
	float gradiants_i_b[2];
	float gradiants_o_w[2];
	float gradiants_o_b;

	for(int i = 0; i < 2; ++i){ // input layer size
		Neuron& neuron = model.input_layer[i];
		for(int j = 0; j < 2; ++j){ // input params size
			neuron.w[j] += direction;
			float dw = compute(model) - computed;
			gradiants_i_w[i][j] = (dw/direction) * learning_rate;
			neuron.w[j] -= direction;
		}
	}
	for(int i = 0; i < 2; ++i){ // input layer size
		Neuron& neuron = model.input_layer[i];
		neuron.b += direction;
		float db = compute(model) - computed;
		gradiants_i_b[i] = (db/direction) * learning_rate;
		neuron.b -= direction;
	}

	for(int i = 0; i < 2; ++i){ // input layer size
		Neuron& neuron = model.output_layer[0];
		neuron.w[i] += direction;
		float dw = compute(model) - computed;
		gradiants_o_w[i] = (dw/direction) * learning_rate;
		neuron.w[i] -= direction;
	}
	Neuron& neuron = model.output_layer[0];
	neuron.b += direction;
	float db = compute(model) - computed;
	gradiants_o_b = (db/direction) * learning_rate;
	neuron.b -= direction;
	neuron.b -= gradiants_o_b;

	for(int i = 0; i < 2; ++i){
		Neuron& neuron = model.input_layer[i];
		for(int j = 0; j < 2; ++j){ // input params size
			neuron.w[j] -= gradiants_i_w[i][j];
		}
	}
	for(int i = 0; i < 2; ++i){
		Neuron& neuron = model.input_layer[i];
		neuron.b -= gradiants_i_b[i];
	}
	for(int i = 0; i < 2; ++i){
		Neuron& neuron = model.output_layer[0];
		neuron.w[i] -= gradiants_o_w[i];
	}
}


void test(const Model& model){
	for(int i = 0; i < TRAINING_SIZE; ++i){
		float a = model.input_layer[0].b + model.input_layer[0].w[0] * train[i][0] + model.input_layer[0].w[1] * train[i][1];
		float b = model.input_layer[1].b + model.input_layer[1].w[0] * train[i][0] + model.input_layer[1].w[1] * train[i][1];
		float actual = model.output_layer[0].b + model.output_layer[0].w[0] * sigmoid(a) + model.output_layer[0].w[1] * sigmoid(b);
		std::cout<<train[i][0]<<" | "<<train[i][1]<<" = "<<sigmoid(actual)<<std::endl;
	}
}
int main(void){
	srand(time(nullptr));

	Model model;

	std::cout<<"w1\t"<<"\t, w2 : "<<"\t,c : "<<std::endl;
	for(int i = 0; i < 1000 * 1000; ++i){
		derive(model);
		if(i % 10000 == 0) 
			std::cout<<"cost : "<<compute(model)<<std::endl;
	}
	test(model);
	std::cout<<std::endl;
	return 0;
}
