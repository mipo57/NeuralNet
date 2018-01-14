/*
 * helpers.h
 *
 *  Created on: 28 gru 2017
 *      Author: mipo57
 */

#ifndef HELPERS_H_
#define HELPERS_H_

#define TEST

#include "stdlib.h"
#include "stdio.h"
#include "types.h"

#define ARR_I (int[])
#define ARR_R (real[])

typedef struct _MemoryChunk {

	void* ptr;
	size_t size;
	struct _MemoryChunk* prev;
} MemoryChunk;

MemoryChunk* last_chunk = NULL;

void* tmalloc(size_t size) {
	void *out = malloc(size);

	if (!out) {
		printf("ERROR: Not enough memory! Exiting...");
		exit(-1);
	}

#ifdef TEST
	MemoryChunk* new_chunk = NULL;
	new_chunk = calloc(1, sizeof(MemoryChunk));
	new_chunk->ptr = out;
	new_chunk->size = size;
	new_chunk->prev = last_chunk;
	last_chunk = new_chunk;

#endif

	return out;
}

void* reallocate(void* ptr, size_t new_size) {
	void *out = realloc(ptr, new_size);

	if (!out) {
		printf("ERROR: Not enough memory! Exiting...");
		exit(-1);
	}

#ifdef TEST
	int ptr_found = 0;
	if (ptr != NULL) {
		MemoryChunk* chunk = last_chunk;
		while(chunk != NULL) {

			if (chunk->ptr == ptr) {

				chunk->size = new_size;
				chunk->ptr = out;
				ptr_found = 1;
				break;
			}
			chunk = chunk->prev;
		}
	}
	else {

		MemoryChunk* new_chunk = NULL;
		new_chunk = calloc(1, sizeof(MemoryChunk));
		new_chunk->ptr = out;
		new_chunk->size = new_size;
		new_chunk->prev = last_chunk;
		last_chunk = new_chunk;
		ptr_found = 1;
	}
	if (!ptr_found)
		printf("Warning: Realloc pointer %p not found\n", ptr);
#endif

	return out;
}

void deallocate(void* ptr) {
	free(ptr);

#ifdef TEST
	MemoryChunk* chunk = last_chunk;
	MemoryChunk* father_chunk = NULL;

	if (ptr == NULL)
		return;

	while(chunk != NULL) {

		if (chunk->ptr == ptr) {

			if (father_chunk != NULL)
				father_chunk->prev = chunk->prev;
			else
				last_chunk = chunk->prev;

			free(chunk);
			return;
		}
		father_chunk = chunk;
		chunk = chunk->prev;
	}

	printf("Warning: Memory %p not found\n", ptr);
#endif
}

int memoryIsEmpty() {

	return last_chunk == NULL;
}

void memoryClear() {

	MemoryChunk *prev, *chunk = last_chunk;
	while(chunk != NULL) {
		prev = chunk->prev;
		free(chunk->ptr);
		free(chunk);
		chunk = prev;
	}

	last_chunk = NULL;
}

size_t memoryGetSize(void *ptr) {

	MemoryChunk *chunk;
	for(chunk = last_chunk; chunk != NULL; chunk = chunk->prev) {

		if (chunk->ptr == ptr)
			return chunk->size;
	}

	return 0;
}

#endif /* HELPERS_H_ */
