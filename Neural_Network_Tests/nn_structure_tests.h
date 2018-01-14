/*
 * nn_structure_tests.h
 *
 *  Created on: 1 sty 2018
 *      Author: mipo57
 */

#ifndef NN_STRUCTURE_TESTS_H_
#define NN_STRUCTURE_TESTS_H_

#include "test_helpers.h"
#include "nn_structure.h"
#include "helpers.h"

int TEST_brainInit() {
	TEST_NAME("brainInit()");

	Brain brain;
	brainInit(&brain);

	ASSERT(PTR_EQUAL(brain.layers, NULL, "Layer vector is empty at the beginning"));
	ASSERT(EQ(brain.num_layers, 0, "Number of layers should be 0"));

	return 1;
}

int TEST_brainAddEmptyLayer() {
	TEST_NAME("brainAddEmptyLayer()");

	Brain brain;
	brainInit(&brain);
	Layer *layer = brainAddEmptyLayer(&brain);

	ASSERT(EQ(brain.num_layers, 1, "Layer count should increase"));
	ASSERT(PTR_NOT_EQUAL(brain.layers, NULL, "Layer vector should exist"));
	ASSERT(PTR_EQUAL(layer, brain.layers, "First layer should be at the beginning of layers vector"));

	ASSERT(PTR_EQUAL(layer->errors.m, NULL, "Errors tensor should be empty"));
	ASSERT(PTR_EQUAL(layer->errors.dims, NULL, "Errors tensor should be empty"));
	ASSERT(PTR_EQUAL(layer->values.m, NULL, "Values tensor should have no values"));
	ASSERT(PTR_EQUAL(layer->values.dims, NULL, "Values tensor should have no dims"));

	ASSERT(EQ(layer->num_parameter_tensors, 0, "There should be no parameters"));
	ASSERT(PTR_EQUAL(layer->parameters, NULL, "There should be no parameters vector"));
	ASSERT(PTR_EQUAL(layer->parameters_delta, NULL, "There should be no parameter deltas vector"));
	ASSERT(EQ(layer->type, empty, "Layer type should be empty"));

	layer = brainAddEmptyLayer(&brain);
	ASSERT(EQ(brain.num_layers, 2, "Layer count should increase"));
	ASSERT(PTR_NOT_EQUAL(brain.layers, NULL, "Layer vector should exist"));
	ASSERT(PTR_EQUAL(layer, brain.layers + 1, "Output should be equal to element of index 1 in layers vector"));

	brainFree(&brain);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}
/* TODO: Create Proper test
int TEST_brainFree() {
	TEST_NAME("brainFree()");

	Brain brain;
	brainAddLayer(&brain);
	brainAddLayer(&brain);

	tensorPlaceholder(brain.layers[0].errors, 2, ARR_I{10, -1});
	tensorPlaceholder(brain.layers[0].values, 2, ARR_I{10, -1});
	brain.layers[0].parameters = NULL;
	brain.layers[0].parameters_delta = NULL;
	brain.layers[0].type = input;
	brain.layers[0].num_parameter_tensors = 0;

	tensorCreate(brain->layers[1].errors, 2, ARR_I{10, 4});
	tensorCreate(brain->layers[1].values, 2, ARR_I{10, 4});
	brain->layers[1].parameters = allocate(sizeof(Tensor) * 2);
	brain->layers[1].parameters_delta = allocate(sizeof(Tensor) * 2);
	tensorCreate(brain->layers[1].errors, 2, ARR_I{10, 4});
	tensorCreate(brain->layers[1].values, 2, ARR_I{10, 4});
	brain->layers[0].type = fc;
	brain->layers[0].num_parameter_tensors = 2;

	brainFree(&brain);
	ASSERT(EQ(brain.num_layers, 0, "Number of layers should be equal to 0"));
	ASSERT(PTR_EQUAL(brain.layers, NULL, "Layer vector should not exist"));

	return 1;
}*/

int TEST_addInputLayer() {
	TEST_NAME("addInputLayer()");

	Brain brain;
	brainInit(&brain);
	addInputLayer(&brain, 30);

	Layer *layer = brain.layers;

	ASSERT(EQ(layer->type, input, "Layer should be of type input"))
	ASSERT(EQ(layer->num_parameter_tensors, 0, "Input layer have no parameters"));
	ASSERT(PTR_EQUAL(layer->parameters, NULL, "Input layer have no parameters"));
	ASSERT(PTR_EQUAL(layer->parameters_delta, NULL, "Input layer have no parameters updates"));
	ASSERT(EQ(layer->values.rank, 2, "Values should be rank 2 tensor (placeholder)"));
	ASSERT(EQ(layer->errors.rank, 2, "Error should be rank 2 tensor (placeholder)"));
	ASSERT(ARRAY_EQ(layer->errors.dims, ARR_I{30, -1}, 2, "Error width should be equal to number of parameters"));
	ASSERT(ARRAY_EQ(layer->values.dims, ARR_I{30, -1}, 2, "Values width should be equal to number of parameters"));

	brainFree(&brain);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_addFullyConnectedLayer() {
	TEST_NAME("addFullyConnectedLayer()");

	Brain brain;
	brainInit(&brain);
	addInputLayer(&brain, 30);
	addFullyConnectedLayer(&brain, 10);

	Layer *layer = brain.layers + 1;

	// Values
	ASSERT(EQ(layer->type, fc, "Layer should be of type fully connected"))
	ASSERT(EQ(layer->values.rank, 2, "Values should be rank 2 tensor (placeholder)"));
	ASSERT(PTR_EQUAL(layer->values.m, NULL, "Values tensor should be only a placeholder"));
	ASSERT(ARRAY_EQ(layer->values.dims, ARR_I{10, -1}, 2, "Values width should be equal to number of neurons"));
	ASSERT(EQ(layer->errors.rank, 2, "Error should be rank 2 tensor (placeholder)"));
	ASSERT(ARRAY_EQ(layer->errors.dims, ARR_I{10, -1}, 2, "Error width should be equal to number of neurons"));
	ASSERT(PTR_EQUAL(layer->errors.m, NULL, "Error should be a placeholder only"));

	// Parameters
	ASSERT(EQ(layer->num_parameter_tensors, 2, "Fully connected layer should have 2 parameters (weight and bias)"));
	ASSERT(PTR_NOT_EQUAL(layer->parameters, NULL, "FC layer have parameters"));
	ASSERT(PTR_NOT_EQUAL(layer->parameters_delta, NULL, "FC layer should have parameters updates"));

	// Weights
	ASSERT(EQ(layer->parameters[0].rank, 2, "Weights rank should be 2"));
	ASSERT(ARRAY_EQ(layer->parameters[0].dims, ARR_I{10, 30}, 2, "Weights size should be num_neurons x num_input"));
	ASSERT(PTR_NOT_EQUAL(layer->parameters[0].m, NULL, "Weights memory should be allocated"));

	// Bias
	ASSERT(EQ(layer->parameters[1].rank, 2, "Bias rank should be 2"));
	ASSERT(ARRAY_EQ(layer->parameters[1].dims, ARR_I{10, 1}, 2, "Bias size should be num_neurosn x 1"));
	ASSERT(PTR_NOT_EQUAL(layer->parameters[1].m, NULL, "Bias memory should be allocated"));

	brainFree(&brain);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_addSigmoidLayer() {
	TEST_NAME("addSigmoidLayer()");

	Brain brain;
	brainInit(&brain);
	addInputLayer(&brain, 20);
	addSigmoidLayer(&brain);

	Layer *layer = brain.layers + 1;

	ASSERT(EQ(layer->type, sigmoid, "It should be a sigmoid layer"));
	ASSERT(PTR_EQUAL(layer->parameters, NULL, "There should be no parameters"));
	ASSERT(PTR_EQUAL(layer->parameters_delta, NULL, "There should be no parameters delta"));
	ASSERT(EQ(layer->values.rank, 2, "Rank of values should be 2"));
	ASSERT(ARRAY_EQ(layer->values.dims, ARR_I{20, -1}, 2, "Values should have width of input and undefined height"));
	ASSERT(PTR_EQUAL(layer->values.m, NULL, "Values should be only a placeholder"));
	ASSERT(EQ(layer->errors.rank, 2, "Rank of errors should be 2"));
	ASSERT(ARRAY_EQ(layer->errors.dims, ARR_I{20, -1}, 2, "Errors should have width of input and undefined height"));
	ASSERT(PTR_EQUAL(layer->errors.m, NULL, "Errors should be only a placeholder"));

	brainFree(&brain);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_allocateMemoryForTraining() {
	TEST_NAME("allocateMemoryForTraining()");

	Brain brain;
	brainInit(&brain);
	addInputLayer(&brain, 20);
	addFullyConnectedLayer(&brain, 15);
	addSigmoidLayer(&brain);

	allocateMemoryForTraining(&brain, 10);

	ASSERT(REAL_ARRAY_LEN_IS(brain.layers[0].values.m, 200, "Input should have size of num_inputs * batch_size"));
	ASSERT(REAL_ARRAY_LEN_IS(brain.layers[0].errors.m, 200, "Input errors should have size of num_inputs * batch_size"));
	ASSERT(REAL_ARRAY_LEN_IS(brain.layers[1].values.m, 150, "FC should have size of num_neurons * batch_size"));
	ASSERT(REAL_ARRAY_LEN_IS(brain.layers[1].errors.m, 150, "FC errors should have size of num_neurons * batch_size"));
	ASSERT(REAL_ARRAY_LEN_IS(brain.layers[2].values.m, 150, "FC should have size of num_layer_inputs * batch_size"));
	ASSERT(REAL_ARRAY_LEN_IS(brain.layers[2].errors.m, 150, "FC errors should have size of num_layer_neurons * batch_size"));

	brainFree(&brain);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_forwardPass() {
	TEST_NAME("forwardPass()");

	Brain brain;
	brainInit(&brain);
	addInputLayer(&brain, 20);
	addFullyConnectedLayer(&brain, 15);
	addSigmoidLayer(&brain);

	allocateMemoryForTraining(&brain, 2);

	real inputs[] = {1.0, 2.0, -3.0, 1.5, 7.0, -10.0};
	forwardPass(&brain, inputs);

	ASSERT(REAL_ARRAY_EQ(brain.layers[0].values.m, ARR_R{1.0, 2.0, -3.0, 1.5, 7.0, -10.0},
				6, 1e-5, "Result of L0 should match precomputed values"));
	ASSERT(REAL_ARRAY_EQ(brain.layers[1].values.m, ARR_R{-6.9, -0.30, -25.5, -7.85},
			4, 1e-5, "Result of L1 should match precomputed values"));
	ASSERT(REAL_ARRAY_EQ(brain.layers[2].values.m, ARR_R{0.00100677082008564, 0.425557483188341,
		8.42346375439769e-12, 0.000389600120838826}, 4, 1e-5, "Result of L2 should match precomputed values"));

	brainFree(&brain);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}
/*
int TEST_forwardPass() {
	TEST_NAME("forwardPass()");

	Brain brain;
	brainInit(&brain);
	addInputLayer(&brain, 20);
	addFullyConnectedLayer(&brain, 15);
	addSigmoidLayer(&brain);

	allocateMemoryForTraining(&brain, 2);

	real inputs[] = {1.0, 2.0, -3.0, 1.5, 7.0, -10.0};
	forwardPass(&brain, inputs);

	real labels[] = {1, 0, 0, 1};
	tensorCreate

	backpropPass(&brain, )

	ASSERT(REAL_ARRAY_EQ(brain.layers[0].values.m, ARR_R{1.0, 2.0, -3.0, 1.5, 7.0, -10.0},
				6, 1e-5, "Result of L0 should match precomputed values"));
	ASSERT(REAL_ARRAY_EQ(brain.layers[1].values.m, ARR_R{-6.9, -0.30, -25.5, -7.85},
			4, 1e-5, "Result of L1 should match precomputed values"));
	ASSERT(REAL_ARRAY_EQ(brain.layers[2].values.m, ARR_R{0.00100677082008564, 0.425557483188341,
		8.42346375439769e-12, 0.000389600120838826}, 4, 1e-5, "Result of L2 should match precomputed values"));

	brainFree(&brain);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}*/

int nn_structure_tests() {

	ASSERT(TEST_brainInit());
	ASSERT(TEST_brainAddEmptyLayer());
	ASSERT(TEST_addInputLayer());
	ASSERT(TEST_addFullyConnectedLayer());
	ASSERT(TEST_addSigmoidLayer());
	ASSERT(TEST_allocateMemoryForTraining());
	ASSERT(TEST_forwardPass());

	return 1;
}

#endif /* NN_STRUCTURE_TESTS_H_ */
