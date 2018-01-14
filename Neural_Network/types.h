/*
 * types.h
 *
 *  Created on: 28 gru 2017
 *      Author: mipo57
 */

#ifndef TYPES_H_
#define TYPES_H_

typedef double real;

typedef struct _Tensor {
	real* m;
	int* dims;
	int rank;
} Tensor;


#endif /* TYPES_H_ */
