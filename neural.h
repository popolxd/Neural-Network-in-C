#ifndef NEURAL_H
#define NEURAL_H

#include "math.h"

void one_propagation_forward(double*** weights, double** biases, double* first_input, int num_of_layers, int* dimensions);

#endif