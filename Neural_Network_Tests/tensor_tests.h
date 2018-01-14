/*
 * tensor_tests.h
 *
 *  Created on: 29 gru 2017
 *      Author: mipo57
 */

#ifndef TENSOR_TESTS_H_
#define TENSOR_TESTS_H_

#include "tensor.h"
#include "test_helpers.h"

int TEST_tensorCreate() {
	TEST_NAME("tensorCreate()");

	Tensor tensor;

	tensorCreate(&tensor, 3, (int[]){5, 3, 7});

	ASSERT(EQ(tensor.rank, 3, "Tensor rank should match desired value"));
	ASSERT(PTR_NOT_EQUAL(tensor.dims, NULL, "Tensor should have dimensions vector"));
	ASSERT(ARRAY_EQ(tensor.dims, (int[]){5,3,7}, 3, "Tensor dimensions should match desired values"));
	ASSERT(PTR_NOT_EQUAL(tensor.m, NULL, "Tensor values vector should exist"));

	deallocate(tensor.dims);
	deallocate(tensor.m);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_tensorEmpty() {
	TEST_NAME("tensorEmpty()");

	Tensor tensor;
	tensorEmpty(&tensor);

	ASSERT(EQ(tensor.rank, -1, "Empty tensor rank should be -1"));
	ASSERT(PTR_EQUAL(tensor.dims, NULL, "Empty tensor dims should not exist"));
	ASSERT(PTR_EQUAL(tensor.m, NULL, "Empty tensor values should not exist"));

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_tensorPlaceholder() {
	TEST_NAME("tensorPlaceholder()");

	Tensor tensor;
	tensorPlaceholder(&tensor, 3, (int[]){2, 3, -1});

	ASSERT(EQ(tensor.rank, 3, "Rank should match"));
	ASSERT(PTR_NOT_EQUAL(tensor.dims, NULL, "Dimensions should exist"));
	ASSERT(ARRAY_EQ(tensor.dims, (int[]){2,3,-1}, 3, "Dimensions should match"));
	ASSERT(PTR_EQUAL(tensor.m, NULL, "Values should not exist"));

	deallocate(tensor.dims);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_tensorFree() {
	TEST_NAME("tensorFree()");

	Tensor tensor;
	tensorCreate(&tensor, 3, (int[]){5, 3, 7});
	tensorFree(&tensor);

	ASSERT(PTR_EQUAL(tensor.m, NULL, "After freeing tensor values should not exist"));
	ASSERT(PTR_EQUAL(tensor.dims, NULL, "After freeing tensor dimensions should not exist"));
	ASSERT(EQ(tensor.rank, -1, "After freeing tensor rank should equal -1"));

	tensorFree(&tensor);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_tensorCreateFromPlaceholder() {
	TEST_NAME("tensorCreateFromPlaceholder()");

	Tensor tensor;
	tensorPlaceholder(&tensor, 5, (int[]){1,-1, 2, 3, -1});
	tensorCreateFromPlaceholder(&tensor, (int[]){2, 3});

	ASSERT(EQ(tensor.rank, 5, "Rank should not change"));
	ASSERT(PTR_NOT_EQUAL(tensor.dims, NULL, "Dimensions should exist"));
	ASSERT(ARRAY_EQ(tensor.dims, (int[]){1,2,2,3,3}, 5, "Missing dimensions should be replaced with provided ones"));
	ASSERT(PTR_NOT_EQUAL(tensor.m, NULL, "Values should exists"));

	tensorFree(&tensor);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_tensorSize() {
	TEST_NAME("tensorSize()");

	Tensor tensor;
	tensorCreate(&tensor, 5, (int[]){3, 7, 1, 3, 2});
	int size = tensorSize(tensor);

	ASSERT(EQ(size, 3 * 7 * 1 * 3 * 2, "Tensor size is product of length of every dimension"));

	tensorFree(&tensor);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_tensorFillValues() {
	TEST_NAME("tensorFillValues()");

	Tensor tensor;
	tensorCreate(&tensor, 3, (int[]){2,2,2});
	real values[] = {1, 2, 3, 4, 5, 6, 7, 8};

	tensorFillValues(&tensor, values);

	ASSERT(REAL_ARRAY_EQ(tensor.m, values, 8, 1e-8, "Tensor values should equal provided ones"));

	tensorFree(&tensor);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int TEST_tensorCreateCopy() {
	TEST_NAME("tensorCreateCopy()");

	Tensor tensor, output;
	tensorCreate(&tensor, 4, (int[]){3, 2, 2, 3});
	real values[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
	tensorFillValues(&tensor, values);

	tensorCreateCopy(tensor, &output);
	ASSERT(EQ(tensor.rank, 4, "After copying tensor ranks should be equal"));
	ASSERT(REAL_ARRAY_EQ(output.m, tensor.m, tensorSize(tensor), 1e-8, "Values should be equal"));
	ASSERT(ARRAY_EQ(output.dims, tensor.dims, 4, "Dimensions should be equal"));
	ASSERT(PTR_NOT_EQUAL(output.m, tensor.m, "Tensors should be independent"));
	ASSERT(PTR_NOT_EQUAL(output.m, tensor.m, "Tensors should be independent"));

	tensorFree(&tensor);
	tensorFree(&output);

	ASSERT(MEMORY_IS_EMPTY());

	return 1;
}

int tensor_test() {

	ASSERT(TEST_tensorCreate());
	ASSERT(TEST_tensorEmpty());
	ASSERT(TEST_tensorPlaceholder());
	ASSERT(TEST_tensorFree());
	ASSERT(TEST_tensorCreateFromPlaceholder());
	ASSERT(TEST_tensorSize());
	ASSERT(TEST_tensorFillValues());
	ASSERT(TEST_tensorCreate());

	return 1;
}

#endif /* TENSOR_TESTS_H_ */
