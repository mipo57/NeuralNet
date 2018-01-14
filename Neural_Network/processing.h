/*
 * processing.h
 *
 *  Created on: 12 sty 2018
 *      Author: mipo57
 */

#ifndef PROCESSING_H_
#define PROCESSING_H_

#include "types.h"

void normalizeData(real* data, int num_entries, int num_data) {

	real* means = calloc(num_entries, sizeof(real));

	int i;
	for (i = 0; i < num_data; i++) {

		int j;
		for (j = 0; j < num_entries; j++) {

			means[j] += data[j + i * num_entries] / num_data;
		}
	}

	for (i = 0; i < num_data; i++) {

		int j;
		for (j = 0; j < num_entries; j++) {

			if (means[j] != 0)
				data[j + i * num_entries] /= means[j];
		}
	}

	free(means);
}

#endif /* PROCESSING_H_ */
