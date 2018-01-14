/*
 * main.c
 *
 *  Created on: 28 gru 2017
 *      Author: mipo57
 */

#define TEST

#include "matrix_tests.h"
#include "nn_common_test.h"
#include "tensor_tests.h"
#include "nn_cost_tests.h"
#include "nn_structure_tests.h"
#include "computation_graph_test.h"

int main() {
	ASSERT(tensor_test());
	ASSERT(matrix_test());
	ASSERT(nn_common_test());
	ASSERT(nn_cost_tests());
	ASSERT(nn_structure_tests());
	ASSERT(computation_graph_test());

	printf("\n##############################\n");
	printf("All tests passed successfully!");
	printf("\n##############################\n");

	return 1;
}
