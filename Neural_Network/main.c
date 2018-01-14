/*
 * main.c
 *
 *  Created on: 12 sty 2018
 *      Author: mipo57
 */

#include "computation_graph.h"
#include "file_io.h"
#include "time.h"
#include "processing.h"

#define NUM_PARAMETERS 16
#define HIDDEN_SIZE 10

int main() {

	srand(123);
	int i;

	real *xs;
	real *ys;
	int num_samples;

	loadTitanicDataset("titanic_training.csv", &xs, &ys, &num_samples);

	Graph graph;
	initGraph(&graph);

	Vertex* inputs = vertexCreateMatrix(NUM_PARAMETERS, -1, 0, 0, 0, &graph);
	Vertex* outputs = vertexCreateMatrix(1, -1, 0, 0, 0, &graph);
	Vertex* wIn = vertexCreateMatrix(HIDDEN_SIZE, NUM_PARAMETERS, 1, 1, 0, &graph);
	Vertex* bIn = vertexCreateMatrix(HIDDEN_SIZE, 1, 1, 1, 0, &graph);

	Vertex* residualWeights[10];
	Vertex* residualBias[10];
	Vertex* residuals[5];

	Vertex* wOut = vertexCreateMatrix(1, HIDDEN_SIZE, 1, 1, 0, &graph);
	Vertex* bOut = vertexCreateMatrix(1, 1, 1, 1, 0, &graph);


	Vertex* l1 = addFullyConnected(inputs, wIn, bIn, &graph);

	for (i = 0; i < 5; i++) {
		residualWeights[2*i]= vertexCreateMatrix(HIDDEN_SIZE, HIDDEN_SIZE, 1, 1, 0, &graph);
		residualBias[2*i] = vertexCreateMatrix(1, HIDDEN_SIZE, 1, 1, 0, &graph);
		residualWeights[2*i+1] = vertexCreateMatrix(HIDDEN_SIZE, HIDDEN_SIZE, 1, 1, 0, &graph);
		residualBias[2*i+1] = vertexCreateMatrix(1, HIDDEN_SIZE, 1, 1, 0, &graph);

		if (i == 0)
			residuals[i] = addResidual(l1, residualWeights[0], residualWeights[1], residualBias[0], residualBias[1], &graph);
		else
			residuals[i] = addResidual(residuals[i-1], residualWeights[2*i], residualWeights[2*i+1], residualBias[2*i], residualBias[2*i+1], &graph);
	}

	Vertex* l3 = addFullyConnected(residuals[4], wOut, bOut, &graph);
	Vertex* cost = addOperationCrossEntropy(l3, outputs, &graph);

	setDims(inputs, ARR_I{NUM_PARAMETERS, num_samples});
	setDims(outputs, ARR_I{1, num_samples});
	reserveMemory(&graph);

	tensorFillValues(&(inputs->values), xs);
	tensorFillValues(&(outputs->values), ys);

	Vertex* weight_updates[4 + 20];
	weight_updates[0] = wIn;
	weight_updates[1] = wOut;
	weight_updates[2] = bIn;
	weight_updates[3] = bOut;


	for (i = 4; i < 14; i++)
		weight_updates[i] = residualWeights[i-4];
	for (i = 14; i < 24; i++)
		weight_updates[i] = residualWeights[i-14];


	initializeRandomly(-0.5, 0.5, weight_updates, 24);

	optimize(cost, weight_updates, 24, NULL, 0, 1e-3, 10000, 1);

	real accuracy = testAccuracy(l3, outputs);
	printf ("Train accuracy: %lf\n", accuracy);

	free(xs);
	free(ys);

	loadTitanicDataset("titanic_test.csv", &xs, &ys, &num_samples);

	freeMemory(&graph, 0);
	setDims(inputs, ARR_I{NUM_PARAMETERS, num_samples});
	setDims(outputs, ARR_I{1, num_samples});
	reserveMemory(&graph);
	tensorFillValues(&inputs->values, xs);
	tensorFillValues(&outputs->values, ys);

	accuracy = testAccuracy(l3, outputs);
	printf ("Test accuracy: %lf\n", accuracy);

	freeGraph(&graph);
	free(xs);
	free(ys);

	int mem = memoryIsEmpty();
	printf("Is memory empty: %d\n", mem);

	return 1;
}
