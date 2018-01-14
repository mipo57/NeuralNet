/*
 * nn_common.h
 *
 *  Created on: 29 gru 2017
 *      Author: mipo57
 */

#ifndef NN_COMMON_H_
#define NN_COMMON_H_

#include "tensor.h"
#include "matrix.h"
#include "math.h"

void layerSigmoid(Tensor input, Tensor* output) {

	int size = tensorSize(input);
	int i;
	for (i = 0; i < size; i++) {
		output->m[i] = 1.0 / (1.0 + exp(-input.m[i]));
	}
}

void layerSigmoidDerv(Tensor values, Tensor errors, Tensor* output) {

	int x, y;
	for (y = 0; y < output->dims[1]; y++) {

		for (x = 0; x < output->dims[0]; x++) {
			real error = errors.m[x + y * errors.dims[0]];
			real value = values.m[x + y * errors.dims[0]];

			output->m[x + y * output->dims[0]] = error * value * (1.0 - value);
		}
	}
}

void layerFullyConnected(Tensor inputs, Tensor weights, Tensor biases, Tensor* outputs) {

	matrixMul(inputs, weights, outputs);
	matrixAddToRows(*outputs, biases, outputs);
}

void layerFullyConnectedDerv(Tensor weights, Tensor errors, Tensor* output) {

	int input_num, batch_num;
	for (batch_num = 0; batch_num < output->dims[1]; batch_num++) {
		
		for (input_num = 0; input_num < output->dims[0]; input_num++) {
			
			real sum = 0;
			int neuron_num;
			for (neuron_num = 0; neuron_num < weights.dims[0]; neuron_num++) {
				
				real error = errors.m[neuron_num + batch_num * errors.dims[0]];
				real weight = weights.m[neuron_num + input_num * weights.dims[0]];
				sum += error * weight;
			}

			output->m[input_num + batch_num * output->dims[0]] = sum;
		}
	}
}

void layerFullyConnectedAvgWeightDerv(Tensor inputs, Tensor errors, Tensor* weight_derv) {

	int neuron_num, input_num, batch_num;

	for (input_num = 0; input_num < weight_derv->dims[1]; input_num++) {

		for (neuron_num = 0; neuron_num < weight_derv->dims[0]; neuron_num++) {

			real sum = 0;
			for (batch_num = 0; batch_num < inputs.dims[1]; batch_num++) {

				real error = errors.m[ neuron_num + batch_num * errors.dims[0] ];
				real input = inputs.m[ input_num + batch_num * inputs.dims[0] ];

				sum += error * input;
			}

			weight_derv->m[neuron_num + input_num * weight_derv->dims[0]] = sum / inputs.dims[1];
		}
	}

}

void clippedParameterUpdate(Tensor* parameter, Tensor updates, real alpha, real threshold) {

	int i, len = tensorSize(updates);

	for (i = 0; i < len; i++) {
		real delta = alpha * updates.m[i];
		parameter->m[i] -= delta < threshold ? delta : (threshold *  (delta > 0 ? 1 : -1));
	}
}

void layerFullyConnectedAvgBiasDerv(Tensor errors, Tensor* bias_derv) {

	int neuron_num, input_num, batch_num;


	for (neuron_num = 0; neuron_num < bias_derv->dims[0]; neuron_num++) {

		real sum = 0;
		for (batch_num = 0; batch_num < errors.dims[1]; batch_num++) {

			real error = errors.m[ neuron_num + batch_num * errors.dims[0] ];
			sum += error;
		}

		bias_derv->m[neuron_num] = sum / errors.dims[1];
	}
}

#endif /* NN_COMMON_H_ */
