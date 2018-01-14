/*
 * matrix.h
 *
 *  Created on: 28 gru 2017
 *      Author: mipo57
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "types.h"
#include "helpers.h"

void matrixCreate(Tensor* tensor, int width, int height) {
	tensor->rank = 2;
	tensor->dims = (int*)tmalloc(sizeof(int) * 2);
	tensor->dims[0] = width;
	tensor->dims[1] = height;
	tensor->m = (real*)tmalloc(sizeof(real) * width * height);
}

void matrixFillValues(Tensor* matrix, real* values) {
	int i = 0;

	if (matrix->rank != 2)
		return;

	for (i = 0; i < matrix->dims[0] * matrix->dims[1]; i++) {
		matrix->m[i] = values[i];
	}
}

void matrixFillIdentity(Tensor* m1) {
	if (m1->dims[0] == m1->dims[1]) {
		int i, j;

		for (int i = 0; i < m1->dims[1]; i++) {
			for (int j = 0; j < m1->dims[0]; j++) {

				int index = j + i * m1->dims[0];
				if (j == i)
					m1->m[index] = 1.0;
				else
					m1->m[index] = 0.0;
			}
		}
	}
}

void matrixMul(Tensor m1, Tensor m2, Tensor* out) {
	int o_width = m2.dims[0];
	int o_height = m1.dims[1];

	if (m1.dims[0] != m2.dims[1]) {
#ifndef TEST
		printf("Warning: matrix dimensions in multiplication don't match");
#endif
		return;
	}

	int x, y;
	for (y = 0; y < o_height; y++) {

		for (x = 0; x < o_width; x++) {

			real sum = 0;
			int i;
			for (i = 0; i < m1.dims[0]; i++) {
				sum += m1.m[i + y * m1.dims[0]] * m2.m[x + i * m2.dims[0]];
			}

			out->m[x + y * o_width] = sum;
		}
	}
}

/*
 * TODO: Needs test
 */
void matrixMulM1Back(Tensor m1, Tensor m2, Tensor errors, Tensor* out) {

	int m1_x, m1_y;
	for (m1_y = 0; m1_y < m1.dims[1]; m1_y++) {

		for (m1_x = 0; m1_x < m1.dims[0]; m1_x++) {

			real sum = 0;
			int m2_x;
			for (m2_x = 0; m2_x < m2.dims[0]; m2_x++) {
				real error = errors.m[m2_x + m1_y * errors.dims[0]];
				sum += m2.m[m2_x + m1_x * m2.dims[0]] * error;
			}

			out->m[m1_x + m1_y * m1.dims[0]] = sum;
		}
	}
}

/*
 * TODO: Needs test
 */
void matrixMulM2Back(Tensor m1, Tensor m2, Tensor errors, Tensor* out) {

	int m2_x, m2_y;
	for (m2_y = 0; m2_y < m2.dims[1]; m2_y++) {

		for (m2_x = 0; m2_x < m2.dims[0]; m2_x++) {

			real sum = 0;
			int m1_y;
			for (m1_y = 0; m1_y < m1.dims[1]; m1_y++) {
				real error = errors.m[m2_x + m1_y * errors.dims[0]];
				sum += m1.m[m2_y + m1_y * m1.dims[0]] * error;
			}

			out->m[m2_x + m2_y * m2.dims[0]] = sum;
		}
	}
}


void matrixTranspose(Tensor matrix, Tensor* out) {

	int x, y;
	for (x = 0; x < matrix.dims[0]; x++) {

		for (y = 0; y < matrix.dims[1]; y++) {

			out->m[y + x*matrix.dims[1]] = matrix.m[x + y*matrix.dims[0]];
		}
	}
}

void matrixAdd(Tensor m1, Tensor m2, Tensor* out) {
	if (m1.dims[0] != m2.dims[0] || m1.dims[1] != m2.dims[1]) {
#ifndef TEST
		printf("Warning: adding matixes of different dimensions");
#endif
		return;
	}

	if (m1.dims[0] != out->dims[0] || m1.dims[1] != out->dims[1]) {
#ifndef TEST
		printf("Warning: Output dimensions are diffrent from those of added matrixes");
#endif
		return;
	}

	int x, y;
	for (y = 0; y < m1.dims[1]; y++) {
		for (x = 0; x < m1.dims[0]; x++) {

			int index = x + y * m1.dims[0];
			out->m[index] = m1.m[index] + m2.m[index];
		}
	}
}

void matrixSub(Tensor m1, Tensor m2, Tensor* out) {
	if (m1.dims[0] != m2.dims[0] || m1.dims[1] != m2.dims[1]) {
#ifndef TEST
		printf("Warning: adding matixes of different dimensions");
#endif
		return;
	}

	if (m1.dims[0] != out->dims[0] || m1.dims[1] != out->dims[1]) {
#ifndef TEST
		printf("Warning: Output dimensions are diffrent from those of added matrixes");
#endif
		return;
	}

	int x, y;
	for (y = 0; y < m1.dims[1]; y++) {
		for (x = 0; x < m1.dims[0]; x++) {

			int index = x + y * m1.dims[0];
			out->m[index] = m1.m[index] - m2.m[index];
		}
	}
}

void matrixMulScalar(Tensor matrix, real scalar, Tensor* out) {

	int x, y;
	for (y = 0; y < matrix.dims[1]; y++) {

		for (x = 0; x < matrix.dims[0]; x++) {

			int index = x + y * matrix.dims[0];
			out->m[index] = scalar * matrix.m[index];
		}
	}
}

void matrixMulBias(Tensor m1, Tensor m2, Tensor* out) {
	int o_width = m2.dims[0];
	int o_height = m1.dims[1];

	if (m1.dims[0] != m2.dims[1] - 1) {
#ifndef TEST
		printf("Warning: matrix dimensions in multiplication with bias don't match");
#endif
		return;
	}


	int x, y;
	for (y = 0; y < o_height; y++) {

		for (x = 0; x < o_width; x++) {

			real sum = 0;
			int i;
			for (i = 0; i < m1.dims[0]; i++) {
				sum += m1.m[i + y * m1.dims[0]] * m2.m[x + i * m2.dims[0]];
			}

			out->m[x + y * o_width] = sum + m2.m[x + m1.dims[0] * m2.dims[0]];
		}
	}
}

void matrixDim1Delta(Tensor m1, Tensor m2, real eps, Tensor* out) {

	int i = 0;
	for (i = 0; i < out->dims[1]; i++) {

		real sum = 0;
		int x;
		for (x = 0; x < m1.dims[0]; x++) {
			sum += (m1.m[x + i * m1.dims[0]] - m2.m[x + i * m2.dims[0]]) / eps;
		}

		out->m[i] = sum;
	}
}

void matrixMulElementwise(Tensor m1, Tensor m2, Tensor* out) {

	int i, size = m1.dims[0] * m1.dims[1];
	for (i = 0; i < size; i++) {

		out->m[i] = m1.m[i] * m2.m[i];
	}
}

void matrixAddToRows(Tensor matrix, Tensor vector, Tensor* output){

	int x, row;
	for (row = 0; row < output->dims[1]; row++) {

		for (x=0; x < output->dims[0]; x++) {

			real matrix_val = matrix.m[x + row * matrix.dims[0]];
			real vector_val = vector.m[x];
			output->m[x + row * output->dims[0]] = matrix_val + vector_val;
		}
	}
}

/*
 * TODO: Write test
 */
void matrixAddToRowsBiasBack(Tensor errors, Tensor* out){

	int out_x, row;
	for (out_x = 0; out_x < out->dims[0]; out_x++) {

		real sum = 0;
		for (row = 0; row < errors.dims[1]; row++)
			sum += errors.m[out_x + row * errors.dims[0]];

		out->m[out_x] = sum;
	}
}

#endif /* MATRIX_H_ */
