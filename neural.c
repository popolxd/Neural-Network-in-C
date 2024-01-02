#include "neural.h"

void one_propagation_forward(weights_struct* weights, biases_struct* biases, double* initial_input, double*** results) {

    // printf("next:\n");

    for (int i = 0; i < weights->main_len; i++) {
        (*results)[i + 1] = malloc(biases->lens[i] * sizeof(double));

        // printf("%p %p\n", (*results)[i], (*results)[i + 1]);

        matrix_vector_mult(weights->weights[i], biases->biases[i], (*results)[i], (*results)[i + 1], weights->lens[i][0], weights->lens[i][1]);
        // printf("\n");
    }

    // printf("next:\n");
    // for (int i = 0; i < (weights->main_len + 1); i++) {
    //     if (i == 0) {
    //         for (int j = 0; j < weights->lens[0][0]; j++) {
    //             printf("%lf\n", (*results)[i][j]);
    //         }
    //     } else {
    //         for (int j = 0; j < biases->lens[biases->main_len - 1]; j++) {
    //             printf("%lf\n", (*results)[i][j]);
    //         }
    //     }
    //    printf("\n");
    // }

    // printf("final result:\n");
    // for (int i = 0; i < biases->lens[biases->main_len - 1]; i++) {
    //     printf("%lf\n", (*results)[weights->main_len][i]);
    // }
}

void last_layer_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double* correct_result, double**** weight_gradients, double*** bias_gradients, double*** layer_a_gradients) {

    for (int i = 0; i < biases->lens[biases->main_len - 1]; i++) {

        double dc_dz;
        // nezalezi, ci pouzijem aktivaciu po relu, alebo pred relu, lebo znamienka ostavaju
        if (results[biases->main_len][i] > 0) {
           dc_dz = 2 * (results[biases->main_len][i] - correct_result[i]);

        } else {
            dc_dz = 2 * (results[biases->main_len][i] - correct_result[i]) * 0.01;
        }

        (*bias_gradients)[biases->main_len - 1][i] += dc_dz;

        for (int j = 0; j < biases->lens[biases->main_len - 2]; j++) {

            (*weight_gradients)[weights->main_len - 1][i][j] += dc_dz * results[biases->main_len - 1][j];
            (*layer_a_gradients)[biases->main_len - 2][j] += dc_dz * weights->weights[weights->main_len - 1][i][j];
        }
    }
}

void one_layer_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double**** weight_gradients, double*** bias_gradients, double*** layer_a_gradients, int normal_layer_order) {

    for (int i = 0; i < biases->lens[biases->main_len - 2 - normal_layer_order]; i++) {

        double dc_dz;

        if (results[biases->main_len - 1 - normal_layer_order][i] > 0) {
            dc_dz = (*layer_a_gradients)[biases->main_len - 2 - normal_layer_order][i];
        } else {
            dc_dz = (*layer_a_gradients)[biases->main_len - 2 - normal_layer_order][i] * 0.01;
        }

        (*bias_gradients)[biases->main_len - 2 - normal_layer_order][i] += dc_dz;

        for (int j = 0; j < biases->lens[biases->main_len - 3 - normal_layer_order]; j++) {

            (*weight_gradients)[weights->main_len - 2 - normal_layer_order][i][j] += dc_dz * results[biases->main_len - 2 - normal_layer_order][j];
            (*layer_a_gradients)[biases->main_len - 3 - normal_layer_order][j] += dc_dz * weights->weights[weights->main_len - 2 - normal_layer_order][i][j];
        }
    }

}

void first_layer_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double**** weight_gradients, double*** bias_gradients, double*** layer_a_gradients) {
    for (int i = 0; i < biases->lens[0]; i++) {

        double dc_dz;

        if (results[1][i] > 0) {
            dc_dz = (*layer_a_gradients)[0][i];
        } else {
            dc_dz = (*layer_a_gradients)[0][i] * 0.01;
        }

        (*bias_gradients)[0][i] += dc_dz;

        for (int j = 0; j < weights->lens[0][0]; j++) {
            (*weight_gradients)[0][i][j] += dc_dz * results[0][j];
        }

    }
}

void one_backpropagation(weights_struct* weights, biases_struct* biases, double** results, double* correct_result, double**** weight_gradients, double*** bias_gradients) { // for now just cleaning malloc

    // for this approach must have at least two layers!
    
    double** layer_a_gradients = malloc((biases->main_len - 1) * sizeof(double*));

    for (int i = 0; i < (biases->main_len - 1); i++) {
        layer_a_gradients[i] = malloc(biases->lens[i] * sizeof(double));
    }
    //

    last_layer_backpropagation(weights, biases, results, correct_result, weight_gradients, bias_gradients, &layer_a_gradients);

    for (int i = 0; i < (biases->main_len - 2); i++) {
        one_layer_backpropagation(weights, biases, results, weight_gradients, bias_gradients, &layer_a_gradients, i);
    }

    first_layer_backpropagation(weights, biases, results, weight_gradients, bias_gradients, &layer_a_gradients);

    // cleaning junk
    for (int i = 0; i < (biases->main_len - 1); i++) {
        free(layer_a_gradients[i]);
    }
    free(layer_a_gradients);
}


void one_training_step(weights_struct* weights, biases_struct* biases, double* initial_input, double* correct_result, double**** weight_gradients, double*** bias_gradients) {

    double** results = malloc((weights->main_len + 1) * sizeof(double*));
    // +1 because I also want to store initial input there for less confusion
    // maybe I will change this later

    // storing initial input
    results[0] = malloc(weights->lens[0][0] * sizeof(double)); // width of first weights matrix

    for (int i = 0; i < weights->lens[0][0]; i++) {
        results[0][i] = initial_input[i];
    }
    //

    one_propagation_forward(weights, biases, initial_input, &results);

    one_backpropagation(weights, biases, results, correct_result, weight_gradients, bias_gradients);

    // cleaning junk
    for (int i = 0; i < (weights->main_len + 1); i++) {
        free(results[i]);
    }
    free(results);
}

void train_neural_network(weights_struct* weights, biases_struct* biases, double learing_rate) { // for loop going trough everything

    // char fen_input[] = "8/3kb3/2p5/3p4/4N3/3K4/8/8 w - - 0 1";
    // double answer[] = {-2.32};
    // double input[385];
    // get_neural_network_input_from_fen(fen_input, input);

    double input[] = {0.348, 1.103, -0.053};
    double answer[] = {2.0, -2.0};

    double*** weight_gradients;
    double** bias_gradients;

    initialize_gradients(&weight_gradients, &bias_gradients, weights, biases);

    for (int i = 0; i < 1000; i++) {
        one_training_step(weights, biases, input, answer, &weight_gradients, &bias_gradients);

        if (i % 10 == 9) {
            adjust_weights_and_biases(weights, biases, &weight_gradients, &bias_gradients, learing_rate);
            set_gradients_to_zero(&weight_gradients, &bias_gradients, weights, biases);
        }
    }

    free_gradients(&weight_gradients, &bias_gradients, weights, biases);
}

void create_neural_network(weights_struct* weights, biases_struct* biases) {
    printf("number of neuron layers: (without input)\n");

    int num_of_neuron_layers;
    scanf("%d", &num_of_neuron_layers);

    int layers[num_of_neuron_layers + 1];

    for (int i = 0; i < (num_of_neuron_layers + 1); i++) {
        if (i == 0) {
            printf("input size: ");
        } else {
            printf("%d. layer: ", i);
        }

        scanf("%d", &layers[i]);
    }

    weights->main_len = num_of_neuron_layers;
    biases->main_len = num_of_neuron_layers;

    weights->weights = malloc(num_of_neuron_layers * sizeof(double**));
    biases->biases = malloc(num_of_neuron_layers * sizeof(double*));

    weights->lens = malloc(num_of_neuron_layers * sizeof(int*));
    biases->lens = malloc(num_of_neuron_layers * sizeof(int));

    for (int i = 0; i < num_of_neuron_layers; i++) {
        weights->lens[i] = malloc(2 * sizeof(int));
        weights->lens[i][0] = layers[i];
        weights->lens[i][1] = layers[i + 1];

        biases->lens[i] = layers[i + 1];

        weights->weights[i] = malloc(layers[i + 1] * sizeof(double*));
        biases->biases[i] = malloc(layers[i + 1] * sizeof(double));

        for (int j = 0; j < layers[i + 1]; j++) {
            weights->weights[i][j] = malloc(layers[i] * sizeof(double));
            biases->biases[i][j] = 0.1; // in relu functions init, otherwise 0

            for (int k = 0; k < layers[i]; k++) {
                weights->weights[i][j][k] = box_muller_normal_distribution() * sqrt(2.0 / (layers[i] + layers[i + 1]));
            }
        }
    }
}

void get_neural_network_input_from_fen(char* fen, double input[385]) {
    int index = 0;
    int fen_index = 0;

    while (fen[fen_index] != ' ') {

        switch(fen[fen_index]) {
            case 'P':
                input[index] = 1;
                for (int i = 1; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'p':
                input[index] = -1;
                for (int i = 1; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'N':
                input[index + 1] = 1;
                for (int i = 2; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'n':
                input[index + 1] = -1;
                for (int i = 2; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'B':
                for (int i = 0; i < 2; i++) input[index + i] = 0;
                input[index + 2] = 1;
                for (int i = 3; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'b':
                for (int i = 0; i < 2; i++) input[index + i] = 0;
                input[index + 2] = -1;
                for (int i = 3; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'R':
                for (int i = 0; i < 3; i++) input[index + i] = 0;
                input[index + 3] = 1;
                for (int i = 4; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'r':
                for (int i = 0; i < 3; i++) input[index + i] = 0;
                input[index + 3] = -1;
                for (int i = 4; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'Q':
                for (int i = 0; i < 4; i++) input[index + i] = 0;
                input[index + 4] = 1;
                for (int i = 5; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'q':
                for (int i = 0; i < 4; i++) input[index + i] = 0;
                input[index + 4] = -1;
                for (int i = 5; i < 6; i++) input[index + i] = 0;
                index += 6;
                break;
            case 'K':
                for (int i = 0; i < 5; i++) input[index + i] = 0;
                input[index + 5] = 1;
                index += 6;
                break;
            case 'k':
                for (int i = 0; i < 5; i++) input[index + i] = 0;
                input[index + 5] = -1;
                index += 6;
                break;
            case '/':
                break;
            default:
                for (int i = 0; i < (fen[fen_index] - 48) * 6; i++) {
                    input[index + i] = 0;
                }
                // printf("%d %c\n", fen[fen_index], fen[fen_index]);
                index += (fen[fen_index] - 48) * 6;
                break;
        }

        fen_index++;
    }
    fen_index++;

    if (fen[fen_index] == 'w') {
        input[index] = 1;
    } else {
        input[index] = 0;
    }
}

void adjust_weights_and_biases(weights_struct* weights, biases_struct* biases, double**** weight_gradients, double*** bias_gradients, double learning_rate) {

    for (int i = 0; i < biases->main_len; i++) {

        for (int j = 0; j < biases->lens[i]; j++) {

            biases->biases[i][j] -= (*bias_gradients)[i][j] * learning_rate;

            for (int k = 0; k < weights->lens[i][0]; k++) {
                weights->weights[i][j][k] -= (*weight_gradients)[i][j][k] * learning_rate;
            }
        }
    }
}

void initialize_gradients(double**** weight_gradients, double*** bias_gradients, weights_struct* weights, biases_struct* biases) {
    *weight_gradients = malloc(weights->main_len * sizeof(double**));
    *bias_gradients = malloc(biases->main_len * sizeof(double*));

    for (int i = 0; i < biases->main_len; i++) {
        (*weight_gradients)[i] = malloc(weights->lens[i][1] * sizeof(double*));
        (*bias_gradients)[i] = malloc(biases->lens[i] * sizeof(double));

        for (int j = 0; j < biases->lens[i]; j++) {
            (*weight_gradients)[i][j] = malloc(weights->lens[i][0] * sizeof(double));
            (*bias_gradients)[i][j] = 0.0;

            for (int k = 0; k < weights->lens[i][0]; k++) {
                (*weight_gradients)[i][j][k] = 0.0;
            }
        }
    }
}


void set_gradients_to_zero(double**** weight_gradients, double*** bias_gradients, weights_struct* weights, biases_struct* biases) {
    for (int i = 0; i < biases->main_len; i++) {
        for (int j = 0; j < biases->lens[i]; j++) {
            (*bias_gradients)[i][j] = 0;

            for (int k = 0; k < weights->lens[i][0]; k++) {
                (*weight_gradients)[i][j][k] = 0;
            }
        }
    }
}

void free_gradients(double**** weight_gradients, double*** bias_gradients, weights_struct* weights, biases_struct* biases) {

    for (int i = 0; i < biases->main_len; i++) {
        for (int j = 0; j < biases->lens[i]; j++) {
            free((*weight_gradients)[i][j]);
        }
        free((*weight_gradients)[i]);
        free((*bias_gradients)[i]);
    }
    free(*weight_gradients);
    free(*bias_gradients);
}