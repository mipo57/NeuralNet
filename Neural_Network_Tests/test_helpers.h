/*
 * helpers.h
 *
 *  Created on: 28 gru 2017
 *      Author: mipo57
 */

#ifndef TEST_HELPERS_H_
#define TEST_HELPERS_H_

#include <math.h>
#include <types.h>
#include <stdio.h>
#include <helpers.h>

#define ASSERT(x) if (!x) return 0;

void TEST_NAME(const char* name) {
	printf("\n%s\n---------------\n", name);
	memoryClear();
}

int MEMORY_IS_EMPTY() {
	if (memoryIsEmpty())
		return 1;
	else
		printf("Memory leak detected!");
	return 0;
}

int REAL_ARRAY_LENGHT(real* array, int expected_len, const char *desc) {

	int real_len = memoryGetSize(array) / sizeof(real);

	if (real_len != expected_len) {
		printf("ERROR in test: %s\nExpected array to have %d elements, got %d\n",  desc, expected_len, real_len);
		return 0;
	}
	else {
		printf("Test passed: %s\n", desc);
		return 1;
	}
}

int DOUBLE_EQ(double actual, double expected, double gran, const char *desc) {
	if (fabsf(expected - actual) > gran) {
		printf("ERROR in test: %s\nExpected %lf, got %lf\n",  desc, expected, actual);
		return 0;
	}
	else {
		printf("Test passed: %s\n", desc);
		return 1;
	}
}

int REAL_ARRAY_EQ(real* actual, real* expected, int len, double gran, const char *desc) {
	int found_error = 0;

	int i;
	for (i = 0; i < len; i++) {
		if (fabsf(expected[i] - actual[i]) > gran) {
			found_error = 1;
			break;
		}
	}

	if (found_error) {
		printf("ERROR in test: %s\nExpected: {", desc);

		for (i = 0; i < len; i++) {
			if (sizeof(real) == sizeof(double))
				printf("%lf, ", expected[i]);
			else
				printf("%f,", expected[i]);
		}
		printf("}\ngot: {");
		for (i = 0; i < len; i++) {
			if (sizeof(real) == sizeof(double))
				printf("%lf, ", actual[i]);
			else
				printf("%f,", actual[i]);
		}
		printf("}\n");
		return 0;
	}
	else {
		printf("Test passed: %s\n", desc);
		return 1;
	}
}


int EQ(int actual, int expected, const char *desc) {
	if (expected != actual) {
		printf("ERROR in test: %s\nExpected %d, got %d\n",  desc, expected, actual);
		return 0;
	}
	else {
		printf("Test passed: %s\n", desc);
		return 1;
	}
}

int ARRAY_EQ(int* actual, int* expected, int len, const char *desc) {
	int found_error = 0;

	int i;
	for (i = 0; i < len; i++) {
		if (expected[i] != actual[i]) {
			found_error = 1;
			break;
		}
	}

	if (found_error) {
		printf("ERROR in test: %s\n Expected: {", desc);

		for (i = 0; i < len; i++) printf("%d, ", expected[i]);;

		printf("}\ngot: {");
		for (i = 0; i < len; i++) printf("%d,", actual[i]);
		printf("}\n");

		return 0;
	}
	else {
		printf("Test passed: %s\n", desc);
		return 1;
	}
}

int ARRAY_LEN_IS(int* array, int expectedLength, const char *desc) {

	int length = memoryGetSize(array) / sizeof(int);

	if (length == expectedLength) {

		printf("Test passed: %s", desc);
		return 1;
	}
	else {

		printf("ERROR in test %s\nExpected array length of %d, got %d", desc, expectedLength, length);
		return 0;
	}
}


int REAL_ARRAY_LEN_IS(real* array, int expectedLength, const char *desc) {

	int length = memoryGetSize(array) / sizeof(real);

	if (length == expectedLength) {

		printf("Test passed: %s\n", desc);
		return 1;
	}
	else {

		printf("ERROR in test %s\nExpected array length of %d, got %d\n", desc, expectedLength, length);
		return 0;
	}
}

int PTR_EQUAL(void *actual, void *expected, const char *desc) {
	if (actual != expected) {
		printf("ERROR in test: %s\nExpected adress to be %p, got %p\n", desc, expected, actual);
		return 0;
	}
	else {
		printf("Test passed: %s\n", desc);
		return 1;
	}
}

int PTR_NOT_EQUAL(void *actual, void *not_expected, const char *desc) {
	if (actual == not_expected) {
		printf("ERROR in test: %s\nExpected adress other than %p\n", desc, not_expected);
		return 0;
	}
	else {
		printf("Test passed: %s\n", desc);
		return 1;
	}
}

#endif /* TEST_HELPERS_H_ */
