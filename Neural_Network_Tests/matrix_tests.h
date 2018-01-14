/*
 * matrix_tests.h
 *
 *  Created on: 28 gru 2017
 *      Author: mipo57
 */

#ifndef MATRIX_TESTS_H_
#define MATRIX_TESTS_H_

#include "matrix.h"
#include "test_helpers.h"
#include "tensor.h"
#include <stdlib.h>

int TEST_createMatrix() {
	TEST_NAME("createMatrix()");

	Tensor tensor;

	matrixCreate(&tensor, 10, 20);

	ASSERT(EQ(tensor.rank, 2, "Rank of matrix is 2"));
	ASSERT(PTR_NOT_EQUAL(tensor.dims, NULL, "Tensor dimensions vector exist"));
	ASSERT(ARRAY_EQ(tensor.dims, (int[2]){10, 20}, 2, "Dimensions are 10x20" ));
	ASSERT(PTR_NOT_EQUAL(tensor.m, NULL, "Tensor values array exist"));

	deallocate(tensor.dims);
	deallocate(tensor.m);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixFillValues() {
	TEST_NAME("matrixFillValues()");

	Tensor tensor;
	matrixCreate(&tensor, 2, 4);

	real values[] = {1, 2, 3, 4, 5, 6, 7, 8};

	matrixFillValues(&tensor, values);

	ASSERT(REAL_ARRAY_EQ(tensor.m, values, 8, 1e-8, "Matrix values should equal provided ones"));

	tensorFree(&tensor);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixFillIdentity() {
	TEST_NAME("matrixFillIdentity");

	Tensor tensor;
	matrixCreate(&tensor, 3, 2);
	real values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	matrixFillValues(&tensor, values);
	matrixFillIdentity(&tensor);

	ASSERT(REAL_ARRAY_EQ(tensor.m, values, 6, 1e-8, "If it is not a square matrix values should not be altered"));
	tensorFree(&tensor);

	matrixCreate(&tensor, 3, 3);
	real identity_values[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	matrixFillIdentity(&tensor);
	ASSERT(REAL_ARRAY_EQ(tensor.m, identity_values, 9, 1e-8, "Matrix values should 1's on diagonal and 0 elsewhere"));
	tensorFree(&tensor);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixMul() {
	TEST_NAME("matrixMul");

	Tensor tensor1, tensor2, tensor3, tensor4, identity;
	matrixCreate(&tensor1, 3, 2);
	matrixCreate(&tensor2, 4, 3);
	matrixCreate(&tensor3, 4, 2);
	matrixCreate(&identity, 3, 3);
	matrixCreate(&tensor4, 3, 2);

	real values1[] = {-1, 5.5, 3,
			           2, -3, 1};
	real values2[] = {1, 2, 3, 4,
			          5, 6, 7, 8,
			          9, 10, 11, 12};
	real values3[] = {53.5, 61, 68.5, 76, -4, -4, -4, -4};
	matrixFillValues(&tensor1, values1);
	matrixFillValues(&tensor2, values2);
	matrixFillIdentity(&identity);

	matrixMul(tensor1, identity, &tensor4);
	ASSERT(REAL_ARRAY_EQ(tensor4.m, values1, 6, 1e-8, "Matrix multiplied by identity matrix should not change"));
	matrixMul(tensor1, tensor2, &tensor3);
	ASSERT(REAL_ARRAY_EQ(tensor3.m, values3, 8, 1e-8, "Matrix multiplication should be consistent with precomputed one"))
	matrixMul(tensor2, tensor1, &tensor3);
	ASSERT(REAL_ARRAY_EQ(tensor3.m, values3, 8, 1e-8, "Matrix multiplication of wrong dimensions don't change output"))

	tensorFree(&tensor2);
	tensorFree(&tensor1);
	tensorFree(&tensor3);
	tensorFree(&tensor4);
	tensorFree(&identity);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixTranspose() {
	TEST_NAME("matrixTranspose");

	Tensor matrix, out;

	matrixCreate(&matrix, 5, 2);
	matrixCreate(&out, 2, 5);

	real values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	real expected_output[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};

	matrixFillValues(&matrix, values);
	matrixTranspose(matrix, &out);

	ASSERT(REAL_ARRAY_EQ(out.m, expected_output, 10, 1e-8, "Transpose should switch matrix elements' indexes"));

	tensorFree(&matrix);
	tensorFree(&out);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}


int TEST_matrixAdd() {
	TEST_NAME("matrixAdd");

	Tensor m1, m2, m3, out;

	matrixCreate(&m1, 3, 2);
	matrixCreate(&m2, 3, 2);
	matrixCreate(&m3, 2, 3);
	matrixCreate(&out, 3, 2);

	real values1[] = {1, 2, 3, 4, 5, 6};
	real values2[] = {3, 5, 1, 45, 55, 16};
	real values3[] = {9, 1, 3, 6 ,1 ,2};
	real values_out[] = {4, 7, 4, 49, 60, 22};

	matrixFillValues(&m1, values1);
	matrixFillValues(&m2, values2);
	matrixFillValues(&m3, values3);

	matrixAdd(m1, m2, &out);
	ASSERT(REAL_ARRAY_EQ(out.m, values_out, 6, 1e-8, "Matrix addition should be equal to adding each components of matrixes"));
	matrixAdd(m1, m3, &out);
	ASSERT(REAL_ARRAY_EQ(out.m, values_out, 6, 1e-8, "Adding matrix of different dimensions should do nothing"));

	tensorFree(&m1);
	tensorFree(&m2);
	tensorFree(&m3);
	tensorFree(&out);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixSub() {
	TEST_NAME("matrixSub");

	Tensor m1, m2, m3, out;

	matrixCreate(&m1, 3, 2);
	matrixCreate(&m2, 3, 2);
	matrixCreate(&m3, 2, 3);
	matrixCreate(&out, 3, 2);

	real values1[] = {1, 2, 3, 4, 5, 6};
	real values2[] = {3, 5, 1, 45, 55, 16};
	real values3[] = {9, 1, 3, 6 ,1 ,2};
	real values_out[] = {-2, -3, 2, -41, -50, -10};

	matrixFillValues(&m1, values1);
	matrixFillValues(&m2, values2);
	matrixFillValues(&m3, values3);

	matrixSub(m1, m2, &out);
	ASSERT(REAL_ARRAY_EQ(out.m, values_out, 6, 1e-8,
			"Matrix subtraction should be equal to subtracting each components of matrixes"));
	matrixSub(m1, m3, &out);
	ASSERT(REAL_ARRAY_EQ(out.m, values_out, 6, 1e-8, "Subtracting matrixes of different dimensions should do nothing"));

	tensorFree(&m1);
	tensorFree(&m2);
	tensorFree(&m3);
	tensorFree(&out);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixMulScalar() {
	TEST_NAME("matrixMulScalar");

	Tensor m1, out;

	matrixCreate(&m1, 3, 2);
	matrixCreate(&out, 3, 2);

	real values1[] = {1, -2, 3, -4, 5, 6};
	real values_out[] = {3, -6, 9, -12, 15, 18};

	matrixFillValues(&m1, values1);

	matrixMulScalar(m1, 3, &out);
	ASSERT(REAL_ARRAY_EQ(out.m, values_out, 6, 1e-8,
			"Multiplying matrix by scalar is multiplying every component of matrix by this scalar"));

	tensorFree(&m1);
	tensorFree(&out);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixMulBias() {
	TEST_NAME("matrixMulBias");

	Tensor tensor1, tensor2, tensor3, tensor4, identity;
	matrixCreate(&tensor1, 3, 2);
	matrixCreate(&tensor2, 4, 4);
	matrixCreate(&tensor3, 4, 2);

	real values1[] = {-1, 5.5, 3,
			           2, -3, 1};
	real values2[] = {1, 2, 3, 4,
			          5, 6, 7, 8,
			          9, 10, 11, 12,
	                  1, 2, 3, 4};
	real values3[] = {54.5, 63, 71.5, 80, -3, -2, -1, 0};

	matrixFillValues(&tensor1, values1);
	matrixFillValues(&tensor2, values2);

	matrixMulBias(tensor1, tensor2, &tensor3);
	ASSERT(REAL_ARRAY_EQ(tensor3.m, values3, 8, 1e-8, "Matrix multiplication with bias should be consistent with precomputed one"))
	matrixMulBias(tensor2, tensor1, &tensor3);
	ASSERT(REAL_ARRAY_EQ(tensor3.m, values3, 8, 1e-8, "Matrix multiplication with bias of wrong dimensions don't change output"))

	tensorFree(&tensor1);
	tensorFree(&tensor2);
	tensorFree(&tensor3);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixDim1Delta() {
	TEST_NAME("matrixDim1Delta()");

	Tensor matrix_max, matrix_min, output;
	real eps = 1e-3;

	real values_max[] = {1, 2, 3, 4, 5, 6};
	real values_min[] = {1.003, 2.003, 4.001, 4.001, 5.004, 6.003};
	real expected_out[] = {-1007, -8};

	matrixCreate(&matrix_max, 3, 2);
	matrixCreate(&matrix_min, 3, 2);
	matrixCreate(&output, 1, 2);
	matrixFillValues(&matrix_max, values_max);
	matrixFillValues(&matrix_min, values_min);

	matrixDim1Delta(matrix_max, matrix_min, eps, &output);

	ASSERT(REAL_ARRAY_EQ(output.m, expected_out, 2, 1e-5, "Deltas should equal those precomputed in sagemath"));

	tensorFree(&matrix_max);
	tensorFree(&matrix_min);
	tensorFree(&output);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixMulElementwise() {
	TEST_NAME("matrixMulElementwise");

	Tensor m1, m2, out;
	matrixCreate(&m1, 3, 2);
	matrixCreate(&m2, 3, 2);
	matrixCreate(&out, 3, 2);

	real v1[] = {1,2,3,4,5,6};
	real v2[] = {3,1,2,6,4,3};
	real expected_out[] = {3, 2, 6, 24, 20, 18};
	matrixFillValues(&m1, v1);
	matrixFillValues(&m2, v2);

	matrixMulElementwise(m1, m2, &out);
	ASSERT(ARRAY_EQ(out.m, expected_out, 6, "Output should be elementwise multiplication of m1 and m2"));
	matrixMulElementwise(m1, m2, &m1);
	ASSERT(ARRAY_EQ(m1.m, expected_out, 6, "Operation should work if first element is also output"));

	tensorFree(&m1);
	tensorFree(&m2);
	tensorFree(&out);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_matrixAddToRows() {
	TEST_NAME("matrixAddToRows()");

	Tensor matrix, vector, outputs;

	matrixCreate(&matrix, 3, 2);
	matrixCreate(&vector, 3, 1);
	matrixCreate(&outputs, 3, 2);

	real matrix_values[] = {1,2,3,4,5,6};
	real vector_values[] = {2,5,7};

	real expected_values[] = {3,7,10,6,10,13};

	matrixFillValues(&matrix, matrix_values);
	matrixFillValues(&vector, vector_values);

	matrixAddToRows(matrix, vector, &outputs);
	ASSERT(REAL_ARRAY_EQ(outputs.m, expected_values, 6, 1e-6, "Function add vector to every row of a matrix"));
	matrixAddToRows(matrix, vector, &matrix);
	ASSERT(REAL_ARRAY_EQ(matrix.m, expected_values, 6, 1e-6, "Function should work if input matrix is also output"));

	tensorFree(&matrix);
	tensorFree(&vector);
	tensorFree(&outputs);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int matrix_test() {
	ASSERT(TEST_createMatrix());
	ASSERT(TEST_matrixFillValues());
	ASSERT(TEST_matrixFillIdentity());
	ASSERT(TEST_matrixMul());
	ASSERT(TEST_matrixTranspose());
	ASSERT(TEST_matrixAdd());
	ASSERT(TEST_matrixSub());
	ASSERT(TEST_matrixMulScalar());
	ASSERT(TEST_matrixMulBias());
	ASSERT(TEST_matrixDim1Delta());
	ASSERT(TEST_matrixMulElementwise());
	ASSERT(TEST_matrixAddToRows());

	return 1;
}

#endif /* MATRIX_TESTS_H_ */
