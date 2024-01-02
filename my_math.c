#include "my_math.h"

float box_muller_normal_distribution() {

    return sqrt(-2 * log((float)rand() / (float)RAND_MAX)) * cos(2*PI*((float)rand() / (float)RAND_MAX));

    // printf("%f\n", r);
}

double vector_dotprod_with_relu(double* vec1, double* vec2, int size, double bias) {

    double result = 0.0;

    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }

    // printf("%lf\n", bias);
    return leaky_relu((result + bias));
}

void matrix_vector_mult(double** matrix, double* biases, double* vec, double* result, int width, int height) {

    for (int i = 0; i < height; i++) {
        // printf("%d", i);
        result[i] = vector_dotprod_with_relu(matrix[i], vec, width, biases[i]);
        // printf("%lf\n", vector_dotprod_with_relu(matrix[i], vec, width, biases[i]));
    }
}

