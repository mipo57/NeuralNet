/*
 * nn_common_test.h
 *
 *  Created on: 29 gru 2017
 *      Author: mipo57
 */

#ifndef NN_COMMON_TEST_H_
#define NN_COMMON_TEST_H_

#include "test_helpers.h"
#include "nn_common.h"
#include "matrix.h"

int TEST_layerSigmoid() {
	TEST_NAME("layerSigmoid()");

	Tensor input, output;
	real inputs[] = {-1, 5.5, 3, 2, -3, 1};
	real expected_output[] = {0.268941421369995,  0.995929862284104,   0.952574126822433,
	                          0.880797077977882,  0.0474258731775668,  0.731058578630005};

	matrixCreate(&input, 3, 2);
	matrixCreate(&output, 3, 2);
	matrixFillValues(&input, inputs);

	layerSigmoid(input, &output);

	ASSERT(REAL_ARRAY_EQ(output.m, expected_output, 6, 1e-5, "Sigmoids values should equal to precomputed ones"));

	tensorFree(&input);
	tensorFree(&output);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_layerSigmoidDerv() {
	TEST_NAME("layerSigmoidDerv()");

	Tensor input_min, input_max, input, output, output_min, output_max, delta, dervs, errors;
	real inputs[] = {-1, 5.5, 3, 2, -3, 1};
	real errors_values[] = {1, 2, 3, 2, 0.3, 0.3};
	real eps = 1e-6;

	matrixCreate(&input, 3, 2);
	matrixCreate(&output_min, 3, 2);
	matrixCreate(&output_max, 3, 2);
	matrixCreate(&output, 3, 2);
	matrixCreate(&errors, 3, 2);
	matrixCreate(&dervs, 3, 2);
	matrixCreate(&delta, 1, 2);
	matrixFillValues(&input, inputs);
	matrixFillValues(&errors, errors_values);

	int i;
	for (i = 0; i < 6; i++) {
		tensorCreateCopy(input, &input_max);
		tensorCreateCopy(input, &input_min);

		input_min.m[i] -= eps;
		input_max.m[i] += eps;

		layerSigmoid(input_min, &output_min);
		layerSigmoid(input_max, &output_max);
		layerSigmoid(input, &output);
		matrixMulElementwise(output_min, errors, &output_min);
		matrixMulElementwise(output_max, errors, &output_max);

		matrixDim1Delta(output_max, output_min, 2 * eps, &delta);
		layerSigmoidDerv(output, errors, &dervs);

		int batch_num = i / 3;

		ASSERT(DOUBLE_EQ(dervs.m[i], delta.m[batch_num], 1e-4, "Derv should be close to delta"));

		tensorFree(&input_max);
		tensorFree(&input_min);
	}

	tensorFree(&input);
	tensorFree(&output);
	tensorFree(&output_max);
	tensorFree(&output_min);
	tensorFree(&delta);
	tensorFree(&dervs);
	tensorFree(&errors);

	ASSERT(MEMORY_IS_EMPTY());

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_layerFullyConnected() {
	TEST_NAME("layerFullyConnected()");

	Tensor inputs, weights, bias, output;
	matrixCreate(&inputs, 3, 2);
	matrixCreate(&weights, 2, 3);
	matrixCreate(&bias, 2, 1);
	matrixCreate(&output, 2, 2);

	real inputs_val[] = {1, -1, 3,
			             5, -0.4, 1.3};
	real weights_val[] = {3, -2, 1.3, 0.4, 0.1, 9};
	real bias_val[] = {3, -5};
	real expected_output[] = {5,     19.6,
	                          17.61, -3.46};
	matrixFillValues(&inputs, inputs_val);
	matrixFillValues(&weights, weights_val);
	matrixFillValues(&bias, bias_val);

	layerFullyConnected(inputs, weights, bias, &output);
	ASSERT(REAL_ARRAY_EQ(output.m, expected_output, 4, 1e-6, "Layer outputs should be equal to ones computed in SAGE"));
	ASSERT(REAL_ARRAY_EQ(inputs.m, inputs_val, 6, 1e-8, "Inputs values should be left unchanged"));
	ASSERT(REAL_ARRAY_EQ(weights.m, weights_val, 6, 1e-8, "Weights values should be left unchanged"));
	ASSERT(REAL_ARRAY_EQ(bias.m, bias_val, 2, 1e-8, "Bias values should be left unchanged"));

	tensorFree(&inputs);
	tensorFree(&bias);
	tensorFree(&weights);
	tensorFree(&output);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_layerFullyConnectedDerv() {
	TEST_NAME("layerFullyConnectedDerv()");

	Tensor weight, bias, input_min, input_max, input, output_min, output_max, delta, dervs, errors;
	real inputs_val[] = {1, -1, 3,
			             5, -0.4, 1.3};
	real weights_val[] = {3, -2, 1.3, 0.4, 0.1, 9};
	real bias_val[] = {3, -5};
	real errors_values[] = {2, 3, 4, 6};
	real eps = 1e-6;

	matrixCreate(&weight, 2, 3);
	matrixCreate(&bias, 2, 1);
	matrixCreate(&input, 3, 2);

	matrixCreate(&output_min, 2, 2);
	matrixCreate(&output_max, 2, 2);

	matrixCreate(&errors, 2, 2);
	matrixCreate(&dervs, 3, 2);
	matrixCreate(&delta, 1, 2);

	matrixFillValues(&input, inputs_val);
	matrixFillValues(&weight, weights_val);
	matrixFillValues(&bias, bias_val);
	matrixFillValues(&errors, errors_values);

	int i;
	for (i = 0; i < 6; i++) {
		tensorCreateCopy(input, &input_max);
		tensorCreateCopy(input, &input_min);

		input_min.m[i] -= eps;
		input_max.m[i] += eps;

		layerFullyConnected(input_min, weight, bias, &output_min);
		layerFullyConnected(input_max, weight, bias, &output_max);

		matrixMulElementwise(output_min, errors, &output_min);
		matrixMulElementwise(output_max, errors, &output_max);

		matrixDim1Delta(output_max, output_min, 2 * eps, &delta);
		layerFullyConnectedDerv(weight, errors, &dervs);

		int batch_num = i / 3;

		ASSERT(DOUBLE_EQ(dervs.m[i], delta.m[batch_num], 1e-4, "Derv should be close to delta"));

		tensorFree(&input_max);
		tensorFree(&input_min);
	}

	tensorFree(&weight);
	tensorFree(&bias);
	tensorFree(&input);
	tensorFree(&output_max);
	tensorFree(&output_min);
	tensorFree(&delta);
	tensorFree(&dervs);
	tensorFree(&errors);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_layerFullyConnectedAvgWeightDerv() {
	TEST_NAME("layerFullyConnectedAvgWeightDerv()");

	Tensor weight, weight_min, weight_max, bias, input, output_min, output_max, delta, dervs, errors;
	real inputs_val[] = {1, -1, 3,
			             5, -0.4, 1.3};
	real weights_val[] = {3, -2, 1.3, 0.4, 0.1, 9};
	real bias_val[] = {3, -5};
	real errors_values[] = {2, 3, 4, 6};
	real eps = 1e-6;

	matrixCreate(&weight, 2, 3);
	matrixCreate(&bias, 2, 1);
	matrixCreate(&input, 3, 2);

	matrixCreate(&output_min, 2, 2);
	matrixCreate(&output_max, 2, 2);

	matrixCreate(&errors, 2, 2);
	matrixCreate(&dervs, 2, 3);
	matrixCreate(&delta, 1, 2);

	matrixFillValues(&input, inputs_val);
	matrixFillValues(&weight, weights_val);
	matrixFillValues(&bias, bias_val);
	matrixFillValues(&errors, errors_values);

	int i;
	for (i = 0; i < 6; i++) {
		tensorCreateCopy(weight, &weight_max);
		tensorCreateCopy(weight, &weight_min);

		weight_min.m[i] -= eps;
		weight_max.m[i] += eps;

		layerFullyConnected(input, weight_min, bias, &output_min);
		layerFullyConnected(input, weight_max, bias, &output_max);

		matrixMulElementwise(output_min, errors, &output_min);
		matrixMulElementwise(output_max, errors, &output_max);

		matrixDim1Delta(output_max, output_min, 2 * eps, &delta);
		layerFullyConnectedAvgWeightDerv(input, errors, &dervs);


		ASSERT(DOUBLE_EQ(dervs.m[i], (delta.m[0] + delta.m[1]) / 2, 1e-4, "Derv should be close to delta"));

		tensorFree(&weight_max);
		tensorFree(&weight_min);
	}

	tensorFree(&weight);
	tensorFree(&bias);
	tensorFree(&input);
	tensorFree(&output_max);
	tensorFree(&output_min);
	tensorFree(&delta);
	tensorFree(&dervs);
	tensorFree(&errors);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_layerFullyConnectedAvgBiasDerv() {
	TEST_NAME("layerFullyConnectedAvgBiasDerv()");

	Tensor weight, bias, bias_min, bias_max, input, output_min, output_max, delta, dervs, errors;
	real inputs_val[] = {1, -1, 3,
			             5, -0.4, 1.3};
	real weights_val[] = {3, -2, 1.3, 0.4, 0.1, 9};
	real bias_val[] = {3, -5};
	real errors_values[] = {2, 3, 4, 6};
	real eps = 1e-6;

	matrixCreate(&weight, 2, 3);
	matrixCreate(&bias, 2, 1);
	matrixCreate(&input, 3, 2);

	matrixCreate(&output_min, 2, 2);
	matrixCreate(&output_max, 2, 2);

	matrixCreate(&errors, 2, 2);
	matrixCreate(&dervs, 2, 1);
	matrixCreate(&delta, 1, 2);

	matrixFillValues(&input, inputs_val);
	matrixFillValues(&weight, weights_val);
	matrixFillValues(&bias, bias_val);
	matrixFillValues(&errors, errors_values);

	int i;
	for (i = 0; i < 2; i++) {
		tensorCreateCopy(weight, &bias_max);
		tensorCreateCopy(weight, &bias_min);

		bias_min.m[i] -= eps;
		bias_max.m[i] += eps;

		layerFullyConnected(input, weight, bias_min, &output_min);
		layerFullyConnected(input, weight, bias_max, &output_max);

		matrixMulElementwise(output_min, errors, &output_min);
		matrixMulElementwise(output_max, errors, &output_max);

		matrixDim1Delta(output_max, output_min, 2 * eps, &delta);
		layerFullyConnectedAvgBiasDerv(errors, &dervs);


		ASSERT(DOUBLE_EQ(dervs.m[i], (delta.m[0] + delta.m[1]) / 2, 1e-4, "Derv should be close to delta"));

		tensorFree(&bias_max);
		tensorFree(&bias_min);
	}

	tensorFree(&weight);
	tensorFree(&bias);
	tensorFree(&input);
	tensorFree(&output_max);
	tensorFree(&output_min);
	tensorFree(&delta);
	tensorFree(&dervs);
	tensorFree(&errors);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int nn_common_test() {
	ASSERT(TEST_layerSigmoid());
	ASSERT(TEST_layerSigmoidDerv());
	ASSERT(TEST_layerFullyConnected());
	ASSERT(TEST_layerFullyConnectedDerv());
	ASSERT(TEST_layerFullyConnectedAvgWeightDerv());
	ASSERT(TEST_layerFullyConnectedAvgBiasDerv());

	return 1;
}

#endif /* NN_COMMON_TEST_H_ */
