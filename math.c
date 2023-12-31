#include "math.h"

double vector_dotprod(double* vec1, double* vec2, int size) {

    double result;

    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

void matrix_vector_mult(double** matrix, double* vec, double* result, int width, int height) {

    for (int i = 0; i < height; i++) {
        result[i] = vector_dotprod(matrix[i], vec, width);
    }
}

