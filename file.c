#include "file.h"

void free_all_bs(weights_struct* weights, biases_struct* biases) {
    // weights
    for (int i = 0; i < weights->main_len; i++) {

        for (int j = 0; j < weights->lens[i][1]; j++) {

            free(weights->weights[i][j]);
        }
        free(weights->lens[i]);
        free(weights->weights[i]);
    }

    free(weights->lens);
    free(weights->weights);

    // biases
    for (int i = 0; i < biases->main_len; i++) {

        free(biases->biases[i]);
    }

    free(biases->lens);
    free(biases->biases);
}

void read_file(char* file_name, weights_struct* weights, biases_struct* biases) {
    FILE* fptr = fopen(file_name, "r");

    if (fptr == NULL) {
        printf("some error happened in reading\n");
        return;
    } else {
        // weights
        int num_of_weights_layers;
        fscanf(fptr, "%d", &num_of_weights_layers);

        weights->main_len = num_of_weights_layers;
        weights->lens = malloc(num_of_weights_layers * sizeof(int*));
        weights->weights = malloc(num_of_weights_layers * sizeof(double**));

        int matrix_width, matrix_height;

        for (int i = 0; i < num_of_weights_layers; i++) {
            fscanf(fptr, "%d %d", &matrix_width, &matrix_height);

            weights->lens[i] = malloc(2 * sizeof(int));
            weights->lens[i][0] = matrix_width;
            weights->lens[i][1] = matrix_height;

            weights->weights[i] = malloc(matrix_height * sizeof(double*));

            for (int j = 0; j < matrix_height; j++) {
                weights->weights[i][j] = malloc(matrix_width * sizeof(double));

                for (int k = 0; k < matrix_width; k++) {
                    double value;
                    fscanf(fptr, "%lf", &value);

                    weights->weights[i][j][k] = value;
                }
            }
        }

        // biases
        int num_of_biases_layers;
        fscanf(fptr, "%d", &num_of_biases_layers);

        biases->main_len = num_of_biases_layers;
        biases->lens = malloc(num_of_biases_layers * sizeof(int));
        biases->biases = malloc(num_of_biases_layers * sizeof(double*));

        int bias_vector_size;

        for (int i = 0; i < num_of_biases_layers; i++) {
            fscanf(fptr, "%d", &bias_vector_size);

            biases->lens[i] = bias_vector_size;
            biases->biases[i] = malloc(bias_vector_size * sizeof(double));

            for (int j = 0; j < bias_vector_size; j++) {
                double value;
                fscanf(fptr, "%lf", &value);

                biases->biases[i][j] = value;
            }
        }

        fclose(fptr);
    }
}

void write_to_file(char* file_name, weights_struct* weights, biases_struct* biases) {
    FILE* fptr = fopen(file_name, "w");

    if (fptr == NULL) {
        printf("some error happened in writing\n");
        return;
    } else {
        char buff[18];

        // weight handeling

        sprintf(buff, "%d\n", weights->main_len);
        fputs(buff, fptr);

        for (int i = 0; i < weights->main_len; i++) {
            sprintf(buff, "%d %d\n", weights->lens[i][0], weights->lens[i][1]); // width height
            fputs(buff, fptr);

            for (int k = 0; k < weights->lens[i][1]; k++) {
                for (int j = 0; j < weights->lens[i][0]; j++) {
                
                    sprintf(buff, "%4.12lf\n", weights->weights[i][k][j]);
                    fputs(buff, fptr);
                }
            }
        }

        // bias handeling

        sprintf(buff, "%d\n", biases->main_len);
        fputs(buff, fptr);
        
        for (int i = 0; i < biases->main_len; i++) {
            sprintf(buff, "%d\n", biases->lens[i]);
            fputs(buff, fptr);

            for (int j = 0; j < biases->lens[i]; j++) {
                sprintf(buff, "%4.12lf\n", biases->biases[i][j]);
                fputs(buff, fptr);
            }
        }

        fclose(fptr);
    }
}