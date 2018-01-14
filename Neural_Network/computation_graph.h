/*
 * computation_graph.h
 *
 *  Created on: 11 sty 2018
 *      Author: mipo57
 */

#ifndef COMPUTATION_GRAPH_H_
#define COMPUTATION_GRAPH_H_

#include "tensor.h"
#include "matrix.h"
#include "nn_common.h"
#include "nn_cost.h"
#include "stdio.h"

typedef enum {e_test, e_variable, e_trainvar, e_input, e_sigmoid, e_mul, e_add_bias, e_crossentropy, e_add} OperationType;

struct _Connection;

typedef struct _Vertex {

	struct _Connection *ingoing;
	int num_ingoing;

	Tensor summed_error;
	Tensor** errors;
	int num_errors;

	Tensor values;
	Tensor values_delta;
	OperationType operation;

	int is_parameter;
	int is_batched;
	int** dimensions;
} Vertex;

typedef struct _Connection {

	Vertex* source;
	Tensor error;
} Connection;

typedef struct _Graph {
	int num_vertexes;
	Vertex** vertexes_list;
	int** dims_list;
	int num_dims;
} Graph;

/*
 * TODO: Write test
 */
void initGraph(Graph* graph) {
	graph->num_vertexes = 0;
	graph->vertexes_list = NULL;
	graph->num_dims = 0;
	graph->dims_list = NULL;
}

void freeMemory(Graph* graph, int clear_parameters) {

	int i;

	for (i = 0; i < graph->num_vertexes; i++) {

		Vertex* vert = graph->vertexes_list[i];

		if (clear_parameters || !vert->is_parameter)  {
			if (!vert->is_batched)
				tensorFree(&vert->values);
			else
				tensorPlaceholderFree(&(vert->values));
		}

		tensorFree(&vert->summed_error);
		int j;
		for (j = 0; j < vert->num_ingoing; j++)
			tensorFree(&vert->ingoing[j].error);

		if (vert->operation == e_trainvar)
			tensorFree(&vert->values_delta);
	}
}

/*
 * TODO: Write test
 */
void addVertexToGraph(Vertex* vertex, Graph* graph) {

	graph->num_vertexes++;
	graph->vertexes_list = reallocate(graph->vertexes_list, graph->num_vertexes * sizeof(Vertex*));
	graph->vertexes_list[graph->num_vertexes - 1] = vertex;
}

/*
 * TODO: Write test
 */
void saveParameters(const char* filename, Vertex** parameter_list, int list_len) {

	FILE* file = fopen(filename, "wb");
	if (!file) {
		printf("Cannot open file: %s!\n", filename);
		return;
	}

	int i = 0;
	for (i = 0; i < list_len; i++) {
		fwrite(parameter_list[i]->values.m, sizeof(real), tensorSize(parameter_list[i]->values), file);
	}

	fclose(file);
}

/*
 * TODO: Write test
 */
void loadParameters(const char* filename, Vertex** parameter_list, int list_len) {

	FILE* file = fopen(filename, "rb");
	if (!file) {
		printf("Cannot open file: %s!\n", filename);
		return;
	}

	int i = 0;
	for (i = 0; i < list_len; i++) {
		fread(parameter_list[i]->values.m, sizeof(real), tensorSize(parameter_list[i]->values), file);
	}

	fclose(file);
}

/*
 * TODO: Write test
 */
void freeGraph(Graph* graph) {

	int i = 0;

	freeMemory(graph, 1);

	for (i = 0; i < graph->num_vertexes; i++) {

		Vertex* vertex = graph->vertexes_list[i];

		int j;
		for (j = 0; j < vertex->num_ingoing; j++)
			tensorFree(&vertex->ingoing[j].error);
		deallocate(graph->vertexes_list[i]->ingoing);

		deallocate(vertex->dimensions);

		tensorFree(&vertex->summed_error);
		tensorFree(&vertex->values);
		tensorFree(&vertex->values_delta);
		deallocate(vertex);
	}
	deallocate(graph->vertexes_list);
	graph->num_vertexes = 0;
	graph->vertexes_list = NULL;

	for (i = 0; i < graph->num_dims; i++)
		deallocate(graph->dims_list[i]);

	deallocate(graph->dims_list);
	graph->dims_list = NULL;
	graph->num_dims = 0;
}

/*
 * TODO: Write test
 */
void vertexConnect(Vertex *source, Vertex *dest) {

	dest->num_ingoing++;
	dest->ingoing = reallocate(dest->ingoing, sizeof(Connection) * dest->num_ingoing);

	Connection* connection = dest->ingoing + dest->num_ingoing - 1;
	connection->source = source;
	tensorEmpty(&connection->error);
}

/*
 * TODO: Write test
 */
Vertex* vertexCreateMatrix(int width, int height, int trainable, int is_parameter, int is_batched, Graph* graph) {

	Vertex* vertex = tmalloc(sizeof(Vertex));
	addVertexToGraph(vertex, graph);

	vertex->ingoing = NULL;
	vertex->num_ingoing = 0;

	tensorEmpty(&vertex->summed_error);
	vertex->errors = NULL;
	vertex->num_errors = 0;

	if (!is_parameter)
		tensorEmpty(&vertex->values);
	else
		tensorCreate(&vertex->values, 2, ARR_I{width, height});

	tensorEmpty(&vertex->values_delta);
	vertex->operation = trainable ? e_trainvar : e_variable;

	int* width_dim = tmalloc(sizeof(int));
	int* height_dim = tmalloc(sizeof(int));

	*width_dim = width;
	*height_dim = height;

	graph->num_dims += 2;
	graph->dims_list = reallocate(graph->dims_list, sizeof(int**) * graph->num_dims);

	graph->dims_list[graph->num_dims - 2] = width_dim;
	graph->dims_list[graph->num_dims - 1] = height_dim;

	vertex->dimensions = tmalloc(sizeof(int*) * 2);
	vertex->dimensions[0] = width_dim;
	vertex->dimensions[1] = height_dim;

	vertex->is_parameter = is_parameter;
	vertex->is_batched = is_batched;

	return vertex;
}

/*
 * TODO: Write test
 */
Vertex* addOperationMatMul(Vertex* m1, Vertex* m2, Graph* graph) {

	Vertex* out = tmalloc(sizeof(Vertex));
	addVertexToGraph(out, graph);

	out->ingoing = tmalloc(sizeof(Vertex) * 2);
	out->num_ingoing = 0;
	vertexConnect(m1, out);
	vertexConnect(m2, out);

	tensorEmpty(&out->summed_error);
	out->errors = NULL;
	out->num_errors = 0;

	tensorEmpty(&out->values);
	tensorEmpty(&out->values_delta);
	out->operation = e_mul;

	out->dimensions = tmalloc(sizeof(int*) * 2);
	out->dimensions[0] = m2->dimensions[0];
	out->dimensions[1] = m1->dimensions[1];

	out->is_parameter = 0;
	out->is_batched = 0;

	return out;

}

Vertex* addOperationAdd(Vertex* m1, Vertex* m2, Graph* graph) {

	Vertex* out = tmalloc(sizeof(Vertex));
	addVertexToGraph(out, graph);

	out->ingoing = tmalloc(sizeof(Vertex) * 2);
	out->num_ingoing = 0;
	vertexConnect(m1, out);
	vertexConnect(m2, out);

	tensorEmpty(&out->summed_error);
	out->errors = NULL;
	out->num_errors = 0;

	tensorEmpty(&out->values);
	tensorEmpty(&out->values_delta);
	out->operation = e_add;

	out->dimensions = tmalloc(sizeof(int*) * 2);
	out->dimensions[0] = m1->dimensions[0];
	out->dimensions[1] = m1->dimensions[1];

	out->is_parameter = 0;
	out->is_batched = 0;

	return out;

}


/*
 * TODO: Write test
 */
Vertex* addOperationBias(Vertex* mat, Vertex* bias, Graph* graph) {

	Vertex* out = tmalloc(sizeof(Vertex));
	addVertexToGraph(out, graph);

	out->ingoing = tmalloc(sizeof(Vertex) * 2);
	out->num_ingoing = 0;
	vertexConnect(mat, out);
	vertexConnect(bias, out);

	tensorEmpty(&out->summed_error);
	out->errors = NULL;
	out->num_errors = 0;

	tensorEmpty(&out->values);
	tensorEmpty(&out->values_delta);
	out->operation = e_add_bias;

	out->dimensions = tmalloc(sizeof(int*) * 2);
	out->dimensions[0] = mat->dimensions[0];
	out->dimensions[1] = mat->dimensions[1];

	out->is_parameter = 0;
	out->is_batched = 0;

	return out;
}

/*
 * TODO: Write test
 */
Vertex* addOperationSigmoid(Vertex* mat, Graph* graph) {

	Vertex* out = tmalloc(sizeof(Vertex));
	addVertexToGraph(out, graph);

	out->ingoing = tmalloc(sizeof(Vertex) * 1);
	out->num_ingoing = 0;
	vertexConnect(mat, out);

	tensorEmpty(&out->summed_error);
	out->errors = NULL;
	out->num_errors = 0;

	tensorEmpty(&out->values);
	tensorEmpty(&out->values_delta);
	out->operation = e_sigmoid;

	out->dimensions = tmalloc(sizeof(int*) * 2);
	out->dimensions[0] = mat->dimensions[0];
	out->dimensions[1] = mat->dimensions[1];

	out->is_parameter = 0;
	out->is_batched = 0;

	return out;
}

/*
 * TODO: Write test
 */
Vertex* addOperationCrossEntropy(Vertex* labels, Vertex* true_labels, Graph* graph) {

	Vertex* out = tmalloc(sizeof(Vertex));
	addVertexToGraph(out, graph);

	out->ingoing = tmalloc(sizeof(Vertex) * 2);
	out->num_ingoing = 0;
	vertexConnect(labels, out);
	vertexConnect(true_labels, out);

	tensorEmpty(&out->summed_error);
	out->errors = NULL;
	out->num_errors = 0;

	tensorEmpty(&out->values);
	tensorEmpty(&out->values_delta);
	out->operation = e_crossentropy;

	graph->num_dims ++;
	graph->dims_list = reallocate(graph->dims_list, graph->num_dims * sizeof(int*));

	int* cr_width = tmalloc(sizeof(int));
	*cr_width = 1;

	graph->dims_list[graph->num_dims - 1] = cr_width;

	out->dimensions = tmalloc(sizeof(int*) * 2);
	out->dimensions[0] = cr_width;
	out->dimensions[1] = labels->dimensions[1];

	out->is_parameter = 0;
	out->is_batched = 0;

	return out;
}

/*
 * TODO: Write test
 */
void reserveMemory(Graph* graph) {

	int i;
	for (i = 0; i < graph->num_vertexes; i++) {

		Vertex* vert = graph->vertexes_list[i];

		if (!vert->is_parameter) {
			if (!vert->is_batched)
				tensorCreate(&vert->values, 2, ARR_I{*(vert->dimensions[0]), *(vert->dimensions[1])});
			else
				tensorPlaceholder(&vert->values, 2, ARR_I{*(vert->dimensions[0]), *(vert->dimensions[1])});
		}


		tensorCreate(&vert->summed_error, 2, ARR_I{*(vert->dimensions[0]), *(vert->dimensions[1])});
		int j;
		for (j = 0; j < vert->num_ingoing; j++) {
			tensorCreate(&vert->ingoing[j].error, 2, ARR_I{*(vert->ingoing[j].source->dimensions[0]),
							*(vert->ingoing[j].source->dimensions[1])});
		}
		if (vert->operation == e_trainvar) {
			tensorCreate(&vert->values_delta, 2, ARR_I{*(vert->dimensions[0]), *(vert->dimensions[1])});
		}
	}
}

/*
 * TODO: Add tests
 */
void setDims(Vertex* vertex, int* dims) {

	int i;
	for (i = 0; i < 2; i++) {

		*(vertex->dimensions[i]) = dims[i];
	}
}

/*
 * TODO: Write test
 */
Vertex* addFullyConnected(Vertex* inputs, Vertex* weights, Vertex* bias, Graph* graph) {

	Vertex* op_mul = addOperationMatMul(inputs, weights, graph);
	Vertex* op_bias = addOperationBias(op_mul, bias, graph);
	Vertex* op_sigmoid = addOperationSigmoid(op_bias, graph);

	return op_sigmoid;
}

Vertex* addResidual(Vertex* inputs, Vertex* w1, Vertex* w2, Vertex* b1, Vertex* b2, Graph* graph) {
	Vertex* l1 = addFullyConnected(inputs, w1, b1, graph);
	Vertex* l2 = addFullyConnected(l1, w2, b2, graph);
	Vertex* out = addOperationAdd(inputs, l2, graph);

	return out;
}

/*
 * TODO: Write test
 */
int isInList(Vertex* vertex, Vertex** list, int list_size) {

	int i;
	for (i = 0; i < list_size; i++) {
		if (list[i] == vertex)
			return 1;
	}

	return 0;
}

/*
 * TODO: Write test
 */
void toplogicalSort(Vertex* vertex, Vertex*** visited_list, int *visited_list_size, Vertex*** stack, int *stack_size) {

	(*visited_list_size) += 1;
	(*visited_list) = reallocate(*visited_list, sizeof(Vertex**) * *visited_list_size);
	(*visited_list)[*visited_list_size - 1] = vertex;

	int i;
	for (i = 0; i < vertex->num_ingoing; i++) {
		Vertex *ingoing = vertex->ingoing[i].source;

		ingoing->num_errors++;
		ingoing->errors = reallocate(ingoing->errors, ingoing->num_errors);
		ingoing->errors[ingoing->num_errors - 1] = &(vertex->ingoing[i].error);

		if (!isInList(ingoing, *visited_list, *visited_list_size)) {

			toplogicalSort(ingoing, visited_list, visited_list_size, stack, stack_size);
		}
	}

	(*stack_size) += 1;
	*stack = reallocate(*stack, sizeof(Vertex**) * *stack_size);
	(*stack)[*stack_size-1] = vertex;
}

/*
 * TODO: Write test
 */
void createComputationList(Vertex* value, Vertex*** o_computation_list, int* o_list_len) {
	*o_computation_list = NULL;
	*o_list_len = 0;

	Vertex** visited_list = tmalloc(sizeof(Vertex*));
	visited_list[0] = value;
	int visited_list_size = 1;

	Tensor* errorTensor = tmalloc(sizeof(Tensor));
	tensorCreate(errorTensor, value->values.rank, value->values.dims);
	tensorFillValue(errorTensor, 1);

	value->errors = tmalloc(sizeof(Tensor*));
	value->num_errors = 1;
	value->errors[0] = errorTensor;

	toplogicalSort(value, &visited_list, &visited_list_size, o_computation_list, o_list_len);
	deallocate(visited_list);
}

/*
 * TODO: Write test
 */
void freeComputationList(Vertex** computation_list, int list_len) {

	int i;
	for (i = 0; i < list_len - 1; i++) {
		deallocate(computation_list[i]->errors);
		computation_list[i]->num_errors = 0;
		computation_list[i]->errors = NULL;
	}

	tensorFree(computation_list[list_len - 1]->errors[0]);
	deallocate(computation_list[list_len - 1]->errors[0]);
	deallocate(computation_list[list_len - 1]->errors);
	deallocate(computation_list);
}

/*
 * TODO: Write test
 */
void executeForwardPass(Vertex** computation_list, int list_len) {

	int i;
	for (i = 0; i < list_len; i++) {

		Vertex* vertex = computation_list[i];

		if (vertex->operation == e_variable)
			continue;
		else if (vertex->operation == e_mul)
			matrixMul(vertex->ingoing[0].source->values, vertex->ingoing[1].source->values, &vertex->values);
		else if (vertex->operation == e_add_bias)
			matrixAddToRows(vertex->ingoing[0].source->values, vertex->ingoing[1].source->values, &vertex->values);
		else if (vertex->operation == e_sigmoid)
			layerSigmoid(vertex->ingoing[0].source->values, &vertex->values);
		else if (vertex->operation == e_crossentropy)
			costCrossEntropy(vertex->ingoing[0].source->values, vertex->ingoing[1].source->values, &vertex->values);
		else if (vertex->operation == e_add)
			matrixAdd(vertex->ingoing[0].source->values, vertex->ingoing[1].source->values, &vertex->values);
	}
}

/*
 * TODO: Write test
 */
void sumErrors(Tensor* error_sum, Tensor** errors, int num_errors) {

	while (num_errors != 1) {
		matrixAdd(*(errors[0]), *(errors[num_errors-1]), errors[0]);
		num_errors--;
	}

	tensorFillValues(error_sum, errors[0]->m);
}

/*
 * TODO: Write test
 */
void executeBackPass(Vertex** computation_list, int list_len) {

	int i;
	for (i = list_len - 1; i >= 0; i--) {

		Vertex* vertex = computation_list[i];

		sumErrors(&vertex->summed_error, vertex->errors, vertex->num_errors);

		if (vertex->operation == e_mul) {

			matrixMulM1Back(vertex->ingoing[0].source->values, vertex->ingoing[1].source->values,
					vertex->summed_error, &(vertex->ingoing[0].error));
			matrixMulM2Back(vertex->ingoing[0].source->values, vertex->ingoing[1].source->values,
							vertex->summed_error, &(vertex->ingoing[1].error));
		}
		else if (vertex->operation == e_trainvar) {

			tensorFillValues(&vertex->values_delta, vertex->summed_error.m);
		}
		else if (vertex->operation == e_add_bias) {

			tensorFillValues(&(vertex->ingoing[0].error), vertex->summed_error.m);
			matrixAddToRowsBiasBack(vertex->summed_error, &(vertex->ingoing[1].error));
		}
		else if (vertex->operation == e_sigmoid) {

			layerSigmoidDerv(vertex->values, vertex->summed_error, &(vertex->ingoing[0].error));
		}
		else if (vertex->operation == e_crossentropy) {

			costCrossEntropyDerv(vertex->ingoing[0].source->values, vertex->ingoing[1].source->values, &(vertex->ingoing[0].error));
			// TODO: Add proper real-value distribution derivative
			tensorFillValue(&(vertex->ingoing[1].error), 0);
		}
		else if (vertex->operation == e_add) {

			tensorFillValues(&(vertex->ingoing[0].error), vertex->summed_error.m);
			tensorFillValues(&(vertex->ingoing[1].error), vertex->summed_error.m);
		}
	}
}

void updateVariables(Vertex** variable_list, int list_len, real alpha) {

	int i;
	for (i = 0; i < list_len; i++) {
		clippedParameterUpdate(&(variable_list[i]->values), variable_list[i]->values_delta, alpha, 1);
	}
}

void optimize(Vertex* value, Vertex** variable_list, int list_len, Vertex** batches_list, int num_batchvars,
		real alpha, int iter, int print_loss) {

	Vertex** computation_list;
	int comp_list_len, i, j;
	createComputationList(value, &computation_list, &comp_list_len);

	for (i = 0; i < iter; i++) {

		for (j = 0; j < num_batchvars; j++)
			tensorPlaceholderNextBatch(&(batches_list[j]->values));

		executeForwardPass(computation_list, comp_list_len);
		executeBackPass(computation_list, comp_list_len);
		updateVariables(variable_list, list_len, alpha);

		if (print_loss) {
			real sum = 0;
			for (j = 0; j < value->values.dims[1]; j++)
				sum += value->values.m[j];
			sum /= value->values.dims[1];
			printf("Iter %d: %lf\n", i, sum);
		}
	}
	freeComputationList(computation_list, comp_list_len);
}

real testAccuracy(Vertex* value, Vertex* true_values) {

	Vertex** computation_list;
	int comp_list_len, i;
	createComputationList(value, &computation_list, &comp_list_len);
	executeForwardPass(computation_list, comp_list_len);

	real accuracy = 0;
	for (i = 0; i < tensorSize(value->values); i++) {

		int prediction = round(value->values.m[i]);
		if (prediction == round(true_values->values.m[i]))
			accuracy += 1;
	}

	accuracy /= tensorSize(value->values);
	freeComputationList(computation_list, comp_list_len);

	return accuracy;
}

void initializeRandomly(real min, real max, Vertex** list, int list_size) {

	int i;
	for (i = 0; i < list_size; i++)
		tensorFillRandom(&(list[i]->values), min, max);
}

#endif /* COMPUTATION_GRAPH_H_ */
