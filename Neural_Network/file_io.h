/*
 * file_io.h
 *
 *  Created on: 12 sty 2018
 *      Author: mipo57
 */

#ifndef FILE_IO_H_
#define FILE_IO_H_

#define BUFFER_SIZE 10000
#define IMG_SIZE 28

// File IO
int getLine(FILE* file, char* buffer) {
	do {
		if (feof(file))
			return 0;

		if (!fgets(buffer, BUFFER_SIZE, file))
			return 0;
	} while (buffer[0] == '#');

	return 1;
}

int loadTitanicDataset(char* path, double **xs, double **ys, int *num_samples) {

	FILE* file = fopen(path, "r");

	if (!file) {
		printf("Nie mozna otworzyc pliku %s!\n", path);
		return 0;
	}

	char buffer[BUFFER_SIZE];

	*xs = NULL;
	*ys = NULL;

	*num_samples = 0;

	getLine(file, buffer);

	while(getLine(file, buffer)) {
		char *label = strtok(buffer, ",");
		int label_num = atoi(label);

		(*num_samples)++;
		*ys = (double*)realloc(*ys, sizeof(double) * 1 * *num_samples);
		*xs = (double*)realloc(*xs, sizeof(double) * 16 * *num_samples);

		// set new ys
		(*ys)[*num_samples - 1] = label_num;

		char *value_str;
		int value_num = 0;
		while( (value_str = strtok(NULL, ",")) ) {
			double value;
			sscanf(value_str, "%lf", &value);

			(*xs)[16 * (*num_samples - 1) + value_num] = value;
			value_num++;
		}
	}

	fclose(file);
	return 1;
}

int loadCSVMnist(char* path, double **xs, double **ys, int *num_samples) {
	FILE* file = fopen(path, "r");

	if (!file) {
		printf("Nie mozna otworzyc pliku %s!\n", path);
		return 0;
	}

	char buffer[BUFFER_SIZE];

	*xs = NULL;
	*ys = NULL;

	*num_samples = 0;

	getLine(file, buffer);

	while(getLine(file, buffer)) {
		char *label = strtok(buffer, ",");
		int label_num = atoi(label);

		(*num_samples)++;
		*ys = (double*)realloc(*ys, sizeof(double) * 10 * *num_samples);
		*xs = (double*)realloc(*xs, sizeof(double) * 784 * *num_samples);

		// set new ys
		int y;
		for (y = 0; y < 10; y++)
			(*ys)[10 * (*num_samples-1) + y] = (y == label_num ? 1.f : 0.f);

		char *value_str;
		int value_num = 0;
		while( (value_str = strtok(NULL, ",")) ) {
			double value = (double)atoi(value_str);

			(*xs)[784 * (*num_samples - 1) + value_num] = value / 1455.f;
			value_num++;
		}
	}

	fclose(file);
	return 1;
}

int loadPGMImage(char* path, double** o_image) {
	FILE* file = fopen(path, "r");

	if (!file) {
		printf("Uwaga: Nie mozna otworzyc pliku %s\n", path);
		return 0;
	}

	char buffer[BUFFER_SIZE];
	getLine(file, buffer); // read header

	int w, h;
	getLine(file, buffer);
	sscanf(buffer, "%d %d", &w, &h);

	if (w != IMG_SIZE && h != IMG_SIZE) {
		printf("Uwaga: Obrazek (%s) ma inne rozmiary niz %dx%d (%dx%d)\n", path, IMG_SIZE, IMG_SIZE, w, h);
		fclose(file);
		return 0;
	}

	int colors;
	getLine(file, buffer);
	sscanf(buffer, "%d %d", &colors);
	if (colors != 255) {
		printf("Maksynalny kolor powinen byc rowny 255, jest %d", colors);
		fclose(file);
		return 0;
	}

	*o_image = (double*)malloc(sizeof(double) * w * h);

	if (!*o_image) {
		printf("Uwaga: Nie mozna zaalokowac pamieci dla obrazka %s\n", path);
		fclose(file);
		return 0;
	}

	int num_read_pixels = 0;
	int tmp_value, local_offset, offset;
	while (getLine(file, buffer)) {
		offset = 0;
		while (sscanf(buffer + offset, " %d %n", &tmp_value, &local_offset) == 1) {
			if (num_read_pixels < w * h) {
				*(*o_image + num_read_pixels) = (double)tmp_value;
				num_read_pixels += 1;
				offset += local_offset;
			}
			else {
				printf("Za duzo pikseli w obrazku %s\n", path);
				free(*o_image);
				fclose(file);
				return 0;
			}
		}
	}

	fclose(file);

	if (num_read_pixels < w * h) {
		printf("Za malo pikseli w obrazku %s\n", path);
		free(*o_image);
		return 0;
	}

	return 1;
}
int savePGMImage(char* path, double* image) {
	FILE* file = fopen(path, "w");
	if (!file) {
		printf("Nie mozna otworzyc pliku %s do zapisu", path);
		return 0;
	}

	fprintf(file, "P2\n");
	fprintf(file, "%d %d\n", IMG_SIZE, IMG_SIZE);
	fprintf(file, "255\n");

	int i;
	for (i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
		fprintf(file, "%d\n", (int)image[i]);
	}

	fclose(file);

	return 1;
}
int loadPPMImage(char* path, double** o_image) {
	FILE* file = fopen(path, "r");

	if (!file) {
		printf("Uwaga: Nie mozna otworzyc pliku %s\n", path);
		return 0;
	}

	char buffer[BUFFER_SIZE];
	getLine(file, buffer); // read header

	int w, h;
	getLine(file, buffer);
	sscanf(buffer, "%d %d", &w, &h);

	if (w != IMG_SIZE && h != IMG_SIZE) {
		printf("Uwaga: Obrazek (%s) ma inne rozmiary niz %dx%d (%dx%d)\n", path, IMG_SIZE, IMG_SIZE, w, h);
		fclose(file);
		return 0;
	}

	int colors;
	getLine(file, buffer);
	sscanf(buffer, "%d %d", &colors);
	if (colors != 255) {
		printf("Maksynalny kolor powinen byc rowny 255, jest %d", colors);
		fclose(file);
		return 0;
	}

	*o_image = (double*)malloc(sizeof(double) * w * h * 3);

	int chanel_jump = w * h;

	if (!*o_image) {
		printf("Uwaga: Nie mozna zaalokowac pamieci dla obrazka %s\n", path);
		fclose(file);
		return 0;
	}

	int num_read_pixels = 0;
	int channel = 0;
	int tmp_value, local_offset, offset;
	while (getLine(file, buffer)) {
		offset = 0;
		while (sscanf(buffer + offset, " %d %n", &tmp_value, &local_offset) == 1) {
			if (num_read_pixels < w * h) {
				*(*o_image + num_read_pixels + chanel_jump * channel) = (double)tmp_value;
				channel++;
				if (channel == 3) {
					num_read_pixels += 1;
					channel = 0;
				}
				offset += local_offset;
			}
			else {
				printf("Za duzo pikseli w obrazku %s\n", path);
				free(*o_image);
				fclose(file);
				return 0;
			}
		}
	}

	fclose(file);

	if (num_read_pixels < w * h) {
		printf("Za malo pikseli w obrazku %s\n", path);
		free(*o_image);
		return 0;
	}

	return 1;
}
int savePPMImage(char* path, double* image) {
	FILE* file = fopen(path, "w");
	if (!file) {
		printf("Nie mozna otworzyc pliku %s do zapisu", path);
		return 0;
	}

	fprintf(file, "P3\n");
	fprintf(file, "%d %d\n", IMG_SIZE, IMG_SIZE);
	fprintf(file, "255\n");

	int channel_jump = IMG_SIZE * IMG_SIZE;

	int i;
	for (i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
		fprintf(file, "%d %d %d\n", (int)image[i], (int)image[i + channel_jump], (int)image[i + 2 * channel_jump]);
	}

	fclose(file);

	return 1;
}

#endif /* FILE_IO_H_ */
