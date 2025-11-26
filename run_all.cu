#include <stdio.h>
#include "mylib.h"
#include "utils.h"


int main() {
    printf("Running hello function:\n");
    hello();
    printf("==========================\n");

    printf("Running device_properties function:\n");
    device_properties();
    printf("==========================\n");

    printf("Running sample_vector_adds function:\n");
    sample_vector_adds();
    printf("==========================\n");

}