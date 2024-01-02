#include "main.h"

int main() {

    biases_struct biases;
    weights_struct weights;

    read_file("first_network.txt", &weights, &biases);
    // create_neural_network(&weights, &biases);

    train_neural_network(&weights, &biases, 0.005);

    // write_to_file("first_network.txt", &weights, &biases);
    free_all_bs(&weights, &biases);

    return 0;
}