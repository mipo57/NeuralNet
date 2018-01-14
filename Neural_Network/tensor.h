/*
 * tensor.h
 *
 *  Created on: 29 gru 2017
 *      Author: mipo57
 */

#ifndef TENSOR_H_
#define TENSOR_H_

#include "types.h"
#include "helpers.h"
#include "string.h"
#include "stdlib.h"

void tensorCreate(Tensor* tensor, int rank, int* dimensions) {

	tensor->rank = rank;
	tensor->dims = tmalloc(sizeof(int) * rank);

	int i, size = 1;
	for (i = 0; i < rank; i++) {
		tensor->dims[i] = dimensions[i];
		size *= dimensions[i];
	}

	tensor->m = tmalloc(sizeof(real) * size);
}



void tensorEmpty(Tensor* tensor) {
	tensor->rank = -1;
	tensor->dims = NULL;
	tensor->m = NULL;
}

void tensorPlaceholder(Tensor* tensor, int rank, int* dims) {

	tensor->rank = rank;
	tensor->m = NULL;
	tensor->dims = tmalloc(sizeof(int) * rank);
	memcpy(tensor->dims, dims, sizeof(int) * rank);

	tensor->data_len = 0;
	tensor->data_pos = 0;
}

void tensorPlaceholderPointData(Tensor* tensor, real* data, int data_len) {

	tensor->m = data;
	tensor->data_len = data_len;
	tensor->data_pos = 0;
}

void tensorPlaceholderNextBatch(Tensor* tensor) {

	int num_data = tensorSize(*tensor);

	if (tensor->data_len - tensor->data_pos - num_data > 0) {
		tensor->m += num_data;
		tensor->data_pos += num_data;
	}
	else {
		tensor->m -= tensor->data_pos;
		tensor->data_pos = 0;
	}
}

void tensorPlaceholderFree(Tensor* tensor) {
	deallocate(tensor->dims);
	tensor->dims = NULL;
	tensor->rank = -1;
	tensor->m = NULL;
	tensor->data_len = 0;
	tensor->data_pos = 0;
}

void tensorCreateFromPlaceholder(Tensor* tensor, int* missing_dims) {

	int i, j = 0, size = 1;
	for (i = 0; i < tensor->rank; i++) {

		if (tensor->dims[i] == -1) {
			tensor->dims[i] = missing_dims[j];
			j++;
		}
		size *= tensor->dims[i];
	}

	tensor->m = tmalloc(sizeof(real) * size);
}

void tensorFree(Tensor* tensor) {

	tensor->rank = -1;
	deallocate(tensor->m);
	deallocate(tensor->dims);

	tensor->m = NULL;
	tensor->dims = NULL;
}

int tensorSize(Tensor tensor) {

	if (tensor.rank < 0)
		return 0;

	int i, size = 1;
	for (i = 0; i < tensor.rank; i++) {
		size *= tensor.dims[i];
	}

	return size;
}

void tensorFillValues(Tensor* tensor, real* values) {

	int size = tensorSize(*tensor);

	memcpy(tensor->m, values, sizeof(real) * size);
}

/*
 * TODO: Write TEST
 */
void tensorFillValue(Tensor* tensor, real value) {

	int size = tensorSize(*tensor);

	int i;
	for (i = 0; i < size; i++)
		tensor->m[i] = value;
}

/*
 * TODO: Write test
 */
void tensorFillRandom(Tensor* tensor, real min, real max) {

	int size = tensorSize(*tensor);

	int i;
	for (i = 0; i < size; i++)
		tensor->m[i] = ((real)rand() / (real)RAND_MAX) * (max - min) + min;
}

void tensorCreateCopy(Tensor source, Tensor* copy) {

	int size = tensorSize(source);

	tensorCreate(copy, source.rank, source.dims);
	memcpy(copy->dims, source.dims, sizeof(int) * source.rank);
	memcpy(copy->m, source.m, sizeof(real) * size);
}

#endif /* TENSOR_H_ */
