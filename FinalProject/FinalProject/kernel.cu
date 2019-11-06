
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

#include <dirent.h>
#include <cufft.h>

#define BLOCK 512

#define SPACE 32
#define DASH 45

#define DEBUG false
#define DIVIDER "############################################\n"

using namespace std;

/*
########################################################################################
########################################################################################
############################### Function signatures ####################################
########################################################################################
########################################################################################
*/

// Cuda kernels
__global__ void _cudaLAD(double *d, cufftComplex* a, cufftComplex* b, int size); // may use shared memory for better movements
__global__ void _cudaCosineSim(double *d, cufftComplex* a, cufftComplex* b, int size); // may use shared memory for better movements
__global__ void _cudaReduceSum(double *d, int size, double *sums);
__global__ void reduction_kernel(double *g_idata, double *g_odata, int size);

// Cuda kernel launchers
double getSimilarityWithCuda(cufftComplex *a, cufftComplex*b, int size);
double getReducedDeviceArray(double *deviceArray, int size);
double reduce(double *arr, int size);

// Cuda helper functions
void checkError(cudaError_t cudaStatus, string message);
cufftComplex* getFourierTransformOfData(int* inp, int begin, int end);

// Serial functions
double cosineSimSerial(cufftComplex *a, cufftComplex *b, int size);
double ladSimSerial(cufftComplex *a, cufftComplex *b, int size);

// General helper functions
void checkSongRead(string path, int index);
int* getSongDataFromFile(string fileName);
vector<string> getSongNames(string songsPath);
cufftComplex* getFourierTransformOfData(int* inp, int begin, int end);

// process functions
int processOnePart(int partIndex, int windowParam);
void processParts(string songsPath, string partsPath, int windowParam);
void testTwoSamples();

/*
########################################################################################
########################################################################################
################################### Main function ######################################
########################################################################################
########################################################################################
*/

int *songsData[100], *partsData[100];
int songsSize;

int main(int argc, char* argv[]) {

	// string songsPath = "C:\\Data\\Songs";
	// string samplesPath = "C:\\Data\\Samples";
	string songsPath = argv[1];
	string samplesPath = argv[2];
	processParts(songsPath, samplesPath, 2);

	// testTwoSamples();

	cudaDeviceReset();
	checkError(cudaGetLastError(), "cudaLastError problem!");

	return EXIT_SUCCESS;

}

/*
########################################################################################
########################################################################################
################################ Process functions #####################################
########################################################################################
########################################################################################
*/

void testTwoSamples() {

	string sample1 = "C:\\Data\\Samples\\sample-day of the lords.txt";
	string sample2 = "C:\\Data\\Samples\\sample-new dawn fades.txt";

	cout << "Reading stage..." << endl;
	int* sData1 = getSongDataFromFile(sample1);
	int* sData2 = getSongDataFromFile(sample2);

	cout << DIVIDER << endl;
	cout << "Calculating for sf1 and sf2" << endl;

	cufftComplex *sf1 = getFourierTransformOfData(sData1, 1, sData1[0] + 1);
	cufftComplex *sf2 = getFourierTransformOfData(sData2, 1, sData2[0] + 1);
	double sim = getSimilarityWithCuda(sf1, sf2, sData1[0]);
	// double sim = cosineSimSerial(sf1, sf2, sData1[0]);

	cout << "Sim: " << sim << endl;
	cout << DIVIDER << endl;
	cout << "Calculating for sf1 and sf1" << endl;

	double simSelf = getSimilarityWithCuda(sf1, sf1, sData1[0]);
	// double simSelf = cosineSimSerial(sf1, sf1, sData1[0]);

	cout << "Sim: " << simSelf << endl;
	cout << DIVIDER << endl;

	cout << "Sim of self: " << simSelf << endl;
	cout << "Sim of diff: " << sim << endl;

}

void processParts(string songsPath, string partsPath, int windowParam) {

	vector<string> songNames = getSongNames(songsPath);
	vector<string> partNames = getSongNames(partsPath);

	songsSize = songNames.size();

#pragma omp parallel for
	for (int i = 0; i < songNames.size(); i++) {
		// if (DEBUG) {
		// 	cout << "Reading song " << songNames[i].c_str() << endl;
		// }
		songsData[i] = getSongDataFromFile(songNames[i]);
		// if (DEBUG) {
		// 	cout << "Read it! " << songsData[i][0] << endl;
		// }
	}

#pragma omp parallel for
	for (int i = 0; i < partNames.size(); i++) {
		// if (DEBUG) {
		// 	cout << "Reading part " << partNames[i].c_str() << endl;
		// }
		partsData[i] = getSongDataFromFile(partNames[i]);
		// if (DEBUG) {
		// 	cout << "Read it! " << partsData[i][0] << endl;
		// }
	}

	for (int i = 0; i < partNames.size(); i++) {
		int bestIndex = processOnePart(i, windowParam);
		cout << partNames[i].c_str() << " ==> " << songNames[bestIndex].c_str() << endl;
	}

	for (int i = 0; i < songsSize; i++) {
		free(songsData[i]);
	}

}

int processOnePart(int partIndex, int windowParam) {

	int *partData = partsData[partIndex];
	int partSize = partData[0];
	int windowMove = partSize / (double)windowParam;
	cufftComplex *partFourier = getFourierTransformOfData(partData, 1, partSize + 1);

	vector<double> songSims;
	for (int i = 0; i < songsSize; i++) {
		int* songData = songsData[i];
		int songSize = songData[0];
		double *partSims = new double[songSize / windowMove + 1];
		for (int w = 0; w < songSize / windowMove + 1; w++) {
			int start = w * windowMove;
			int end = start + partSize;
			if (start >= songSize) {
				break;
			}
			cufftComplex *songWindowFourier;
			if (end < songSize) {
				songWindowFourier = getFourierTransformOfData(songData, start + 1, end + 1);
			}
			else {
				songWindowFourier = getFourierTransformOfData(songData, songSize - partSize + 1, songSize + 1);
			}
			partSims[w] = getSimilarityWithCuda(partFourier, songWindowFourier, partSize);
			free(songWindowFourier);
		}
		songSims.push_back(*min_element(partSims, partSims + (songSize / windowMove + 1)));
	}

	free(partData);

	int bestSongIndex = distance(songSims.begin(), min_element(songSims.begin(), songSims.end()));

	// if (DEBUG) {
	// 	cout << "Best similarity: " << songSims[bestSongIndex] << endl;
	// }
	return bestSongIndex;

}

/*
########################################################################################
########################################################################################
################################# Serial functions #####################################
########################################################################################
########################################################################################
*/

double dot(cufftComplex a, cufftComplex b) {
	return a.x * b.x + a.y * b.y;
}

double cosineSimSerial(cufftComplex *a, cufftComplex *b, int size) {
	double cosines = 0.0;
	for (int i = 0; i < size; i++) {
		double ab = dot(a[i], b[i]);
		double aa = dot(a[i], a[i]);
		double bb = dot(b[i], b[i]);
		cosines += abs(ab / sqrt(aa * bb));
	}
	return cosines / size;
}

double ladSimSerial(cufftComplex *a, cufftComplex *b, int size) {
	double lad = 0.0;
	for (int i = 0; i < size; i++) {
		double x = sqrt(a[i].x * a[i].x + a[i].y * a[i].y);
		double y = sqrt(b[i].x * b[i].x + b[i].y * b[i].y);
		lad += abs(x - y);
	}
	return lad;
}

/*
########################################################################################
########################################################################################
############################## Cuda helper functions ###################################
########################################################################################
########################################################################################
*/

void checkError(cudaError_t cudaStatus, string message) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s\n", message);
		exit(1);
	}
}

// Exclusive
cufftComplex* getFourierTransformOfData(int* inp, int begin, int end) {

	cufftHandle plan;
	cufftComplex *data;
	cudaError_t cudaStatus;

	size_t partSize = (end - begin) * sizeof(cufftComplex);

	cufftComplex *transformed = (cufftComplex *)malloc(partSize);
	cudaStatus = cudaMalloc((void**)&data, partSize);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for fourier transform!\n");
		goto Error;
	}

#pragma omp parallel for
	for (int i = 0; i < end - begin; i++) {
		transformed[i].x = inp[i + begin];
		transformed[i].y = 0;
	}

	cudaStatus = cudaMemcpy(data, transformed, partSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed before fourier transofrm!\n");
		goto Error;
	}

	if (cufftPlan1d(&plan, end - begin, CUFFT_C2C, 1) != CUFFT_SUCCESS) { // Todo: check batch, set to 1 here
		fprintf(stderr, "CUFFT error: Plan creation failed!\n");
		goto Error;
	}

	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed!\n");
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(transformed, data, partSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed after fourier transofrm!\n");
		goto Error;
	}

	// if (DEBUG) {
	// 	for (int i = 0; i < end - begin; i++) {
	// 		printf("T[%d] = (%lf, %lf)\n", i, transformed[i].x, transformed[i].y);
	// 	}
	// }

	cufftDestroy(plan);
	cudaFree(data);
	return transformed;

Error:
	cufftDestroy(plan);
	free(transformed);
	cudaFree(data);
	return NULL;

}

/*
########################################################################################
########################################################################################
################################### CUDA kernels #######################################
########################################################################################
########################################################################################
*/

__global__ void reduction_kernel(double *g_idata, double *g_odata, int size) {
	__shared__ double sdata[BLOCK];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

__global__ void _cudaCosineSim(double *d, cufftComplex* a, cufftComplex* b, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double ab, aa, bb;
	if (i < size) {
		ab = a[i].x * b[i].x + a[i].y * b[i].y;
		aa = a[i].x * a[i].x + a[i].y * a[i].y;
		bb = b[i].x * b[i].x + b[i].y * b[i].y;
		if (aa * bb == 0) {
			d[i] = 1;
		}
		else {
			d[i] = abs(ab / sqrt(aa * bb));
		}
	}
}

__global__ void _cudaLAD(double *d, cufftComplex* a, cufftComplex* b, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double x, y;
	if (i < size) {
		x = sqrt(a[i].x * a[i].x + a[i].y * a[i].y);
		y = sqrt(b[i].x * b[i].x + b[i].y * b[i].y);
		d[i] = abs(x - y);
	}
}

/*
########################################################################################
########################################################################################
################################## CUDA launchers ######################################
########################################################################################
########################################################################################
*/

double getSimilarityWithCuda(cufftComplex *a, cufftComplex*b, int size) {

	cudaError_t cudaStatus;
	double *deviceO;
	cufftComplex *deviceA, *deviceB;
	size_t partSize = size * sizeof(cufftComplex);

	cudaStatus = cudaMalloc((void**)&deviceA, partSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for first malloc!\n");
		goto Error2;
	}

	cudaStatus = cudaMalloc((void**)&deviceB, partSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for second malloc!\n");
		goto Error2;
	}

	cudaStatus = cudaMalloc((void**)&deviceO, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for deviceD!\n");
		goto Error2;
	}

	cudaStatus = cudaMemcpy(deviceA, a, partSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "first cudaMemcpy failed before fourier transofrm!\n");
		goto Error2;
	}

	cudaStatus = cudaMemcpy(deviceB, b, partSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "second cudaMemcpy failed before fourier transofrm!\n");
		goto Error2;
	}

	int blocks = ((size - 1) / BLOCK) + 1;
	_cudaLAD << <blocks, BLOCK >> > (deviceO, deviceA, deviceB, size);

	checkError(cudaDeviceSynchronize(), "Can not synchronize after sim kernel!");

	double *out = (double*)malloc(sizeof(double) * size);
	checkError(cudaMemcpy(out, deviceO, sizeof(double) * size, cudaMemcpyDeviceToHost), "Out copy for sim kernel");
	double result = reduce(out, size);

	// NOTE (/ size for cosine sim)
	result = result;

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceO);
	return result;

Error2:
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceO);
	return -1.0;

}

int level = 0;
double *d_array[1000];

double reduce(double *arr, int size) {

	int newsize = pow(2, ceil(log(size) / log(2)));
	arr = (double *)realloc(arr, newsize * sizeof(double));
	for (int i = size; i < newsize; i++)
		arr[i] = 0;

	cudaError_t status;

	int i = newsize;
	int dcount = 0;
	while (i != 0) {
		status = cudaMalloc((void **)&d_array[level], i * sizeof(double));
		//printf("allocated level %d with size %d\n", level, i);
		dcount++;
		checkError(status, "cudaMalloc(d_array[level])");

		if (i == 1) {
			i = 0;
		}
		else {
			i = ((i - 1) / BLOCK) + 1;
		}

		level++;
	}

	status = cudaMemcpy(d_array[0], arr, newsize * sizeof(double), cudaMemcpyHostToDevice);
	checkError(status, "Memcpy(d_array[0], arr)");

	int current = newsize;
	int next = ((current - 1) / BLOCK) + 1;
	int counter = 0;

	while (current != 1) {
		reduction_kernel << <next, BLOCK >> > (d_array[counter], d_array[counter + 1], current);
		//printf("called kernel for level %d and %d\n", counter, counter+1);
		checkError(cudaGetLastError(), "kernel");
		current = next;
		next = ((current - 1) / BLOCK) + 1;
		counter++;
	}

	status = cudaDeviceSynchronize();
	checkError(status, "Dev Sync");
	cudaMemcpy(arr, d_array[level - 1], sizeof(double), cudaMemcpyDeviceToHost);

	for (int j = 0; i < dcount; i++) {
		status = cudaFree(d_array[i]);
		checkError(status, "cudaFree");
	}
	level = 0;

	double r = arr[0];
	free(arr);
	return r;
}

double getReducedDeviceArray(double *deviceArray, int size) {

	// int blocks = ((size-1) / BLOCK) + 1;
	// double *sums, *deviceSums;

	// sums = (double *) malloc(sizeof(double) * blocks);
	// checkError(cudaMalloc((void**)&deviceSums, sizeof(double) * blocks), "cudaMalloc in getReduced...");

	// _cudaReduceSum<<<blocks, BLOCK>>>(deviceArray, size, deviceSums);

	// checkError(cudaMemcpy(sums, deviceSums, sizeof(double) * blocks, cudaMemcpyDeviceToHost), "cudaMemcpy back to host");

	// double result = 0.0;
	// for (int i = 0; i < blocks; i++) {
	// 	result += sums[i];
	// }
	// return result;

	return -1.0;

}

/*
########################################################################################
########################################################################################
############################# General helper functions #################################
########################################################################################
########################################################################################
*/

void checkSongRead(string path, int index) {
	// 	if (songsRead[index] > 0) {
	// 		return;
	// 	}
	// 	int *data = getSongDataFromFile(path);
	// 	if (songsRead[index] > 0) {
	// 		return;
	// 	}
	// 	// #pragma omp atomic {
	// 		if (songsData.size() <= index) {
	// 			songsData.push_back(data);
	// 		} else {
	// 			songsData[index] = data;
	// 		}
	// 		songsRead[index] += 1;
	// 	// }
}

vector<string> getSongNames(string songsPath) {

	vector<string> songNames;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(songsPath.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			string fname = ent->d_name;
			if (fname != "." && fname != "..") {
				songNames.push_back(songsPath + "\\" + ent->d_name);
			}
		}
		closedir(dir);
		return songNames;
	}
	else {
		/* could not open directory */
		perror("");
		return vector<string>();
	}

}

int* getSongDataFromFile(string fileName) {

	FILE* fp = fopen(fileName.c_str(), "r");
	fseek(fp, 0L, SEEK_END);
	int fileSize = ftell(fp);
	fseek(fp, 0L, SEEK_SET);

	int* filedata = (int *)malloc(sizeof(int) * (fileSize + 1));
	char c;
	int j = 1;
	bool negative = false;
	filedata[1] = 0;

	while ((c = getc(fp)) != EOF) {
		int d = c;
		if (d != SPACE) {
			if (d == DASH) {
				negative = true;
			}
			else {
				filedata[j] += d - 48;
				filedata[j] *= 10;
			}
		}
		else {
			if (negative) {
				filedata[j] /= -10;
			}
			else {
				filedata[j] /= 10;
			}
			negative = false;
			j++;
			filedata[j] = 0;
		}
	}
	if (negative) {
		filedata[j] /= -10;
	}
	else {
		filedata[j] /= 10;
	}
	fclose(fp);

	filedata[0] = j;
	return filedata;

}
