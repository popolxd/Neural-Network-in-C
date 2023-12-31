#ifndef FILE_H
#define FILE_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double** biases;
    int main_len;
    int* lens;
} biases_struct;

typedef struct {
    double*** weights;
    int main_len;
    int** lens;
} weights_struct;

void free_all_bs(weights_struct* weights, biases_struct* biases);
void read_file(char* file_name, weights_struct* weights, biases_struct* biases);
void write_to_file(char* file_name, weights_struct* weights, biases_struct* biases);

#endif