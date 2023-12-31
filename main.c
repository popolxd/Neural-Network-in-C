#include "main.h"

int main() {

    biases_struct biases;
    weights_struct weights;

    read_file("first_network.txt", &weights, &biases);
    write_to_file("first_network.txt", &weights, &biases);
    free_all_bs(&weights, &biases);

    return 0;
}