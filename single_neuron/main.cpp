#include <cmath>
#include <cstdlib>
#include <iostream>
#include <time.h>


float train[][3] = {
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 1}
};
#define SIZE (int)(sizeof(train)/sizeof(train[0]))


float rand_float(void);

struct Neuron {
    float z[2];
    float b;

	Neuron(){
		for(int i = 0; i < 2; ++i){
			this->z[i] = rand_float();
		}
		this->b = rand_float();
	}
};
// generate a random float between 0 and 1
float rand_float(void){
	return (float) rand()/(float) RAND_MAX;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}
float compute(Neuron& neuron){

	float result = 0.0f;
	for(int i = 0; i < SIZE; ++i){
		float actual = neuron.b;
		for(int j = 0; j < 2; ++j){
			actual += neuron.z[j] * train[i][j];
		}
		float expected = train[i][2];
		float loss = (expected - sigmoid(actual));
		result += loss * loss;
	}
	result /= (float)SIZE;
	return result;
}

Neuron derive(Neuron& neuron, float direction = 0.0001f, float learning_rate = 1){
	float computed = compute(neuron);
	float gradiants[3];
	for(int i = 0; i < 2; ++i){
		neuron.z[i] += direction;
		float dz = compute(neuron) - computed;
		gradiants[i] = (dz/direction) * learning_rate;
		neuron.z[i] -= direction;
	}

	neuron.b += direction;
	float db = compute(neuron) - computed;
	gradiants[2] = (db/direction) * learning_rate;
	neuron.b -= direction;
	for(int i = 0; i < 2; ++i){
		neuron.z[i] -= gradiants[i];
	}
	neuron.b -= gradiants[2];
	return neuron;
}

void test(Neuron& neuron){
	for(int i = 0; i < SIZE; ++i){
		float actual = neuron.b;
		for(int j = 0; j < 2; ++j){
			actual += neuron.z[j] * train[i][j];
		}
		std::cout<<train[i][0]<<" | "<<train[i][1]<<" = "<<sigmoid(actual)<<std::endl;
	}
}
int main(void){
	srand(time(NULL));

	Neuron neuron;

	std::cout<<"w1"<<"\t, w2 : "<<"\t,c : "<<std::endl;
	for(int i = 0; i < 100 * 2; ++i){
		neuron = derive(neuron);
	}
	test(neuron);
	std::cout<<std::endl;
	return 0;
}
