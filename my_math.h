#ifndef MATH_H
#define MATH_H

#define leaky_relu(x) ((x > 0) ? x : 0.01*x)
#define PI 3.1415926536

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double vector_dotprod_with_relu(double* vec1, double* vec2, int size, double bias);
void matrix_vector_mult(double** matrix, double* biases, double* vec, double* result, int width, int height);
float box_muller_normal_distribution();

#endif