/*
 * computation_graph_test.h
 *
 *  Created on: 11 sty 2018
 *      Author: mipo57
 */

#ifndef COMPUTATION_GRAPH_TEST_H_
#define COMPUTATION_GRAPH_TEST_H_

#include "computation_graph.h"
#include "test_helpers.h"

int TEST_createMatrixPlaceholder() {
	TEST_NAME("createMatrixPlaceholder()");

	Element *el = createMatrixPlaceholder(10, -1, 1);

	ASSERT(PTR_NOT_EQUAL(el, NULL, "Function output should exist"));

	ASSERT(EQ(el->num_input_nodes, 0, "After creation there should be no input nodes"));
	ASSERT(PTR_EQUAL(el->in_nodes, NULL, "Input nodes should be null vector"));

	ASSERT(EQ(el->values.rank, 2, "Values rank should be 2"));
	ASSERT(ARRAY_EQ(el->values.dims, ARR_I{10, -1}, 2, "Dimensions should be equal to input"));
	ASSERT(PTR_EQUAL(el->values.m, NULL, "Values should be only a placeholder"));

	ASSERT(PTR_EQUAL(el->errors, NULL, "There should be no error connections"));

	ASSERT(EQ(el->values_delta.rank, 2, "Delta rank should exist and be equal 2"));
	ASSERT(ARRAY_EQ(el->values_delta.dims, ARR_I{10, -1}, 2, "Deltas dimensions should be equal to input"));
	ASSERT(PTR_EQUAL(el->values_delta.m, NULL, "Deltas should be only a placeholder"));

	ASSERT(EQ(el->operation, e_variable, "Element type should be variable"));

	elementFree(&el);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_addOperationToGraph() {
	TEST_NAME("addOperationToGraph()");

	Element *e1, *e2, *e3;
	e1 = createMatrixPlaceholder(20, -1, 1);
	e2 = createMatrixPlaceholder(10, -1, 1);
	e3 = createMatrixPlaceholder(30, -1, 0);

	Element* test_op = addOperationToGraph((Element[]){e1, e2, e3}, num_inputs, e_test);

	ASSERT(EQ(test_op->num_input_nodes, 3, "Operation should count for every input node"));
	ASSERT(PTR_NOT_EQUAL(test_op->in_nodes, NULL, "Input nodes array should exist"));
	ASSERT(PTR_EQUAL(test_op->in_nodes[0], e1, "Elements in input connections should match provided ones"));
	ASSERT(PTR_EQUAL(test_op->in_nodes[1], e2, "Elements in input connections should match provided ones"));
	ASSERT(PTR_EQUAL(test_op->in_nodes[2], e3, "Elements in input connections should match provided ones"));

	// TODO: Continue Test

	return 1;
}

int computation_graph_test() {
	ASSERT(TEST_createMatrixPlaceholder());

	return 1;
}

#endif /* COMPUTATION_GRAPH_TEST_H_ */
