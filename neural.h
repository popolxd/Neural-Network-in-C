#ifndef NEURAL_H
#define NEURAL_H

#include "my_math.h"
#include "file.h"

#include <stdio.h>
#include <math.h>

void one_propagation_forward(weights_struct* weights, biases_struct* biases, double* initial_input, double*** results);

void last_layer_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double* correct_result, double**** weight_gradients, double*** bias_gradients, double*** layer_a_gradients);
void one_layer_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double**** weight_gradients, double*** bias_gradients, double*** layer_a_gradients, int normal_layer_order);
void first_layer_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double**** weight_gradients, double*** bias_gradients, double*** layer_a_gradients);

void one_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double* correct_result, double**** weight_gradients, double*** bias_gradients);

void one_training_step(weights_struct* weights, biases_struct* biases, double* initial_input, double* correct_result, double**** weight_gradients, double*** bias_gradients);
void train_neural_network(weights_struct* weights, biases_struct* biases, double learing_rate);

void create_neural_network(weights_struct* weights, biases_struct* biases);
void get_neural_network_input_from_fen(char* fen, double input[385]);

void adjust_weights_and_biases(weights_struct* weights, biases_struct* biases, double**** weight_gradients, double*** bias_gradients, double learing_rate);

void initialize_gradients(double**** weight_gradients, double*** bias_gradients, weights_struct* weights, biases_struct* biases);
void set_gradients_to_zero(double**** weight_gradients, double*** bias_gradients, weights_struct* weights, biases_struct* biases);
void free_gradients(double**** weight_gradients, double*** bias_gradients, weights_struct* weights, biases_struct* biases);

#endif