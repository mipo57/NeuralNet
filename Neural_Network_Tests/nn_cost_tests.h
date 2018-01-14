/*
 * nn_cost_tests.h
 *
 *  Created on: 30 gru 2017
 *      Author: mipo57
 */

#ifndef NN_COST_TESTS_H_
#define NN_COST_TESTS_H_

#include "nn_cost.h"
#include "matrix.h"

int TEST_costCrossEntropy() {
	TEST_NAME("costCrossEntropy()");

	Tensor labels, true_labels, cross_entropy;

	matrixCreate(&labels, 3, 2);
	matrixCreate(&true_labels, 3, 1);
	matrixCreate(&cross_entropy, 1, 2);

	real label_values[] = {0.34, 0.78, 0.13, 0.56, 0.24, 0.89};
	real true_values[] = {0, 1, 1};
	real expected_val[] = {0.901399210595573, 0.788210241321976};

	matrixFillValues(&labels, label_values);
	matrixFillValues(&true_labels, true_values);

	costCrossEntropy(labels, true_labels, &cross_entropy);

	ASSERT(REAL_ARRAY_EQ(cross_entropy.m, expected_val, 2, 1e-6, "Cross entropies should equal those computed in SAGE"))

	tensorFree(&labels);
	tensorFree(&true_labels);
	tensorFree(&cross_entropy);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_costCrossEntropyDerv() {
	TEST_NAME("costCrossEntropyDerv()");

	Tensor labels, labels_max, labels_min, true_labels, output_max, output_min, derv, delta;
	real eps = 1e-6;

	matrixCreate(&labels, 3, 2);
	matrixCreate(&true_labels, 3, 1);
	matrixCreate(&output_min, 1, 2);
	matrixCreate(&output_max, 1, 2);
	matrixCreate(&delta, 1, 2);
	matrixCreate(&derv, 3, 2);

	real label_values[] = {0.34, 0.78, 0.13, 0.56, 0.24, 0.89};
	real true_values[] = {0, 1, 1};

	matrixFillValues(&labels, label_values);
	matrixFillValues(&true_labels, true_values);

	int i;
	for (i = 0; i < 6; i++) {
		tensorCreateCopy(labels, &labels_max);
		tensorCreateCopy(labels, &labels_min);

		labels_max.m[i] += eps;
		labels_min.m[i] -= eps;

		costCrossEntropy(labels_min, true_labels, &output_min);
		costCrossEntropy(labels_max, true_labels, &output_max);

		matrixDim1Delta(output_max, output_min, 2 * eps, &delta);
		costCrossEntropyDerv(labels, true_labels, &derv);

		int batch_num = i / 3;
		ASSERT(DOUBLE_EQ(derv.m[i], delta.m[batch_num], 1e-4, "Derivative should be close to delta"));

		tensorFree(&labels_max);
		tensorFree(&labels_min);
	}

	tensorFree(&labels);
	tensorFree(&true_labels);
	tensorFree(&output_min);
	tensorFree(&output_max);
	tensorFree(&delta);
	tensorFree(&derv);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int nn_cost_tests() {
	ASSERT(TEST_costCrossEntropy());
	ASSERT(TEST_costCrossEntropyDerv());

	return 1;
}

#endif /* NN_COST_TESTS_H_ */
