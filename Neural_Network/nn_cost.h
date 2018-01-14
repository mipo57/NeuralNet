/*
 * nn_cost.h
 *
 *  Created on: 30 gru 2017
 *      Author: mipo57
 */

#ifndef NN_COST_H_
#define NN_COST_H_

void costCrossEntropy(Tensor labels, Tensor true_labels, Tensor* out) {

	int label_num, batch_num;

	for (batch_num = 0; batch_num < out->dims[1]; batch_num++) {

		double cross_entropy = 0;
		for (label_num = 0; label_num < labels.dims[0]; label_num++) {

			real true_lab = true_labels.m[label_num + batch_num * true_labels.dims[0]];
			real label = labels.m[label_num + batch_num * labels.dims[0]];

			cross_entropy += true_lab * log(label + 1e-7) + (1 - true_lab) * log(1 - label + 1e-7);
		}

		out->m[batch_num] = -cross_entropy / labels.dims[0];
	}
}

void costCrossEntropyDerv(Tensor labels, Tensor true_labels, Tensor* derv) {

	int label_num, batch_num;
	for (batch_num = 0; batch_num < derv->dims[1]; batch_num++) {

		for (label_num = 0; label_num < derv->dims[0]; label_num++) {

			real true_lab = true_labels.m[label_num + batch_num * true_labels.dims[0]];
			real label = labels.m[label_num + batch_num * labels.dims[0]];

			derv->m[label_num + batch_num * derv->dims[0]] = -(true_lab / (label + 1e-7) - (1 - true_lab)/(1-label + 1e-7))/labels.dims[0];
		}
	}
}

#endif /* NN_COST_H_ */
