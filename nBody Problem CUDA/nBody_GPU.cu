//GPU code goes here
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
Running without arguments is equivalent to 1000 iterations with the
5 celestial objects declared in the golden_bodies array.

$ nbody.exe 1000 5

The output of this shows the energy before and after the simulation,
and should be:

-0.169075164
-0.169087605
*/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

using type = double;

const type pi{ 3.141592653589793 };
const type solar_mass{ 4 * pi * pi };
const type days_per_year{ 365.24 };

int blockSize;
int minGridSize;
int gridSize;

template <typename T>
struct planet {
	T x, y, z;
	T vx, vy, vz;
	T mass;
};

template <typename T>
void advance(int nbodies, planet<T> *bodies)
{
	int i, j;

	for (i = 0; i < nbodies; ++i) {
		planet<T> &b = bodies[i];

		for (j = i + 1; j < nbodies; j++) {
			planet<T> &b2 = bodies[j];
			T dx = b.x - b2.x;
			T dy = b.y - b2.y;
			T dz = b.z - b2.z;
			T inv_distance = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
			T mag = inv_distance * inv_distance * inv_distance;
			b.vx -= dx * b2.mass * mag;
			b.vy -= dy * b2.mass * mag;
			b.vz -= dz * b2.mass * mag;
			b2.vx += dx * b.mass  * mag;
			b2.vy += dy * b.mass  * mag;
			b2.vz += dz * b.mass  * mag;
		}
	}

	for (i = 0; i < nbodies; ++i) {
		planet<T> &b = bodies[i];
		b.x += b.vx;
		b.y += b.vy;
		b.z += b.vz;
	}
}

template<typename T>
__device__ void warpReduce(volatile T *sdata, unsigned int tID)
{
	sdata[tID] += sdata[tID + 32];
	sdata[tID] += sdata[tID + 16];
	sdata[tID] += sdata[tID + 8];
	sdata[tID] += sdata[tID + 4];
	sdata[tID] += sdata[tID + 2];
	sdata[tID] += sdata[tID + 1];
}

template <typename T>
__global__ void energyKernel(int nbodies, T *addReduc, T *subReduc, planet<T> *bodies){
	extern __shared__ T e[];

	unsigned int tID = threadIdx.x;
	unsigned int i = tID + blockIdx.x * (blockDim.x * 2);

	if ((i + blockDim.x) < nbodies){
		planet<T> &b = bodies[i];
		planet<T> &b2 = bodies[i + blockDim.x];
		
		e[tID] = (0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz)) + (0.5 * b2.mass * (b2.vx * b2.vx + b2.vy * b2.vy + b2.vz * b2.vz));
	}
	else if(i < nbodies)
	{
		planet<T> &b = bodies[i];
		e[tID] = 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
	}

	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if (tID < stride)
		{
			e[tID] += e[tID + stride];
		}
		__syncthreads();
	}

	if (tID < 32){ warpReduce(e, tID); }

	if (tID == 0)
	{
		addReduc[blockIdx.x] = e[0];
	}

	__syncthreads();

	//------------------------------------------------

	//int savedi = 1;
	//for (int i = 1, j = 0; i < nbodies; i++)
	//{
	//	planet<T> &b = bodies[j];
	//	planet<T> &b2 = bodies[i];
	//	T dx = b.x - b2.x;
	//	T dy = b.y - b2.y;
	//	T dz = b.z - b2.z;
	//	T distance = sqrt(dx * dx + dy * dy + dz * dz);
	//	e -= (b.mass * b2.mass) / distance;

	//	if (i == nbodies - 1 && savedi < nbodies)
	//	{
	//		i = savedi;
	//		savedi = i + 1;
	//		j++;
	//	}
	//}
	e[tID] = 0;

	/*if ((i + blockDim.x) < nbodies)
	{
		for (int iter = i + 1; iter < nbodies - blockDim.x; iter++){
			planet<T> &b = bodies[i];
			planet<T> &b2 = bodies[iter];
			T dx = b.x - b2.x;
			T dy = b.y - b2.y;
			T dz = b.z - b2.z;
			T distance = sqrt(dx * dx + dy * dy + dz * dz);

			planet<T> &b3 = bodies[iter + blockDim.x];
			dx = b.x - b3.x;
			dy = b.y - b3.y;
			dz = b.z - b3.z;
			T distance2 = sqrt(dx * dx + dy * dy + dz * dz);

			e[tID] += ((b.mass * b2.mass) / distance) + ((b.mass * b3.mass) / distance2);
		}
	}*/
	/*else*/ if(i < nbodies)
	{
		for (int iter = i + 1; iter < nbodies; iter++){
			planet<T> &b = bodies[i];
			planet<T> &b2 = bodies[iter];
			T dx = b.x - b2.x;
			T dy = b.y - b2.y;
			T dz = b.z - b2.z;
			T distance = sqrt(dx * dx + dy * dy + dz * dz);
			e[tID] += (b.mass * b2.mass) / distance;
		}
	}

	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if (tID < stride)
		{
			e[tID] += e[tID + stride];
		}
		__syncthreads();


	}

	if (tID < 32){ warpReduce(e, tID); }

	if (tID == 0)
	{
		subReduc[blockIdx.x] = e[0];
	}
}

template <typename T>
T energy(int nbodies, planet<T> *bodies)
{
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, energyKernel<type>, 0, nbodies);
	gridSize = (nbodies + blockSize - 1) / blockSize;

	if (gridSize > 1){
		gridSize = (int)ceil((float)gridSize / 2);
	}
	else{
		blockSize = (int)ceil((float)blockSize / 2);
	}

	T *h_addArray = new T[gridSize];
	T *h_subArray = new T[gridSize];

	T *d_addArray; cudaMalloc((void**)&d_addArray, gridSize * sizeof(T));
	T *d_subArray; cudaMalloc((void**)&d_subArray, gridSize * sizeof(T));

	planet<T> *d_bodies; cudaMalloc((void**)&d_bodies, nbodies * sizeof(planet<T>));

	cudaMemcpy(d_bodies, bodies, nbodies * sizeof(planet<T>), cudaMemcpyHostToDevice);
	energyKernel << <gridSize, blockSize, nbodies * sizeof(planet<T>) >> >(nbodies, d_addArray, d_subArray, d_bodies);
	cudaMemcpy(h_addArray, d_addArray, gridSize * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_subArray, d_subArray, gridSize * sizeof(T), cudaMemcpyDeviceToHost);

	for (int i = 1; i < gridSize; i++)
	{
		h_addArray[0] += h_addArray[i];
		h_subArray[0] += h_subArray[i];
	}

	T e = h_addArray[0] - h_subArray[0];
	
	return e;
}


//template <typename T>
//T energy(int nbodies, planet<T> *bodies) {
//	T e = 0.0;
//	T e2 = 0.0;
//
//	//GPU
//	/*for (int i = 0; i < nbodies; ++i) {
//		planet<T> &b = bodies[i];
//		e += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
//		//test += 5;
//
//		for (int j = i + 1; j < nbodies; j++) {
//			planet<T> &b2 = bodies[j];
//			T dx = b.x - b2.x;
//			T dy = b.y - b2.y;
//			T dz = b.z - b2.z;
//			T distance = sqrt(dx * dx + dy * dy + dz * dz);
//			e -= (b.mass * b2.mass) / distance;
//			//test -= 1;
//		}
//	}*/
//
//	for (int i = 0; i < nbodies; i++)
//	{
//		planet<T> &b = bodies[i];
//		e += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
//	}
//
//	int savedi = 1;
//	for (int i = 1, j = 0; i < nbodies; i++)
//	{
//		planet<T> &b = bodies[j];
//		planet<T> &b2 = bodies[i];
//		T dx = b.x - b2.x;
//		T dy = b.y - b2.y;
//		T dz = b.z - b2.z;
//		T distance = sqrt(dx * dx + dy * dy + dz * dz);
//		e2 += (b.mass * b2.mass) / distance;
//
//		if (i == nbodies - 1 && savedi < nbodies)
//		{
//			i = savedi;
//			savedi = i+1;
//			j++;
//		}
//	}
//
//	T total = e - e2;
//	int m = 3;
//
//	return e;
//}

template<typename T>
__global__ void reduceSum(planet<T> *bodies, T *outdata, int arrayIdent, int nbodies){
	extern __shared__ T sdata[];

	if (arrayIdent == 1){
		unsigned int tID = threadIdx.x;
		unsigned int i = tID + blockIdx.x * (blockDim.x * 2);

		if (tID < nbodies && (i + blockDim.x) < nbodies){
			sdata[tID] = (bodies[i].vx * bodies[i].mass) + (bodies[i + blockDim.x].vx * bodies[i + blockDim.x].mass);
		}
		else{
			sdata[tID] = (bodies[i].vx * bodies[i].mass);
		}

		__syncthreads();

		for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
		{
			if (tID < stride)
			{
				sdata[tID] += sdata[tID + stride];
			}
			__syncthreads();


		}

		if (tID < 32){ warpReduce(sdata, tID); }

		if (tID == 0)
		{
			outdata[blockIdx.x] = sdata[0];
		}
	}

	if (arrayIdent == 2){
		unsigned int tID = threadIdx.x;
		unsigned int i = tID + blockIdx.x * (blockDim.x * 2);
		if (tID < nbodies && (i + blockDim.x) < nbodies){
			sdata[tID] = (bodies[i].vy * bodies[i].mass) + (bodies[i + blockDim.x].vy * bodies[i + blockDim.x].mass);
		}
		else{
			sdata[tID] = (bodies[i].vy * bodies[i].mass);
		}
		__syncthreads();

		for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
		{
			if (tID < stride)
			{
				sdata[tID] += sdata[tID + stride];
			}
			__syncthreads();
		}

		if (tID < 32){ warpReduce(sdata, tID); }

		if (tID == 0)
		{
			outdata[blockIdx.x] = sdata[0];
		}
	}

	if (arrayIdent == 3){
		unsigned int tID = threadIdx.x;
		unsigned int i = tID + blockIdx.x * (blockDim.x * 2);
		if (tID < nbodies && (i + blockDim.x) < nbodies){
			sdata[tID] = (bodies[i].vz * bodies[i].mass) + (bodies[i + blockDim.x].vz * bodies[i + blockDim.x].mass);
		}
		else{
			sdata[tID] = (bodies[i].vz * bodies[i].mass);
		}
		__syncthreads();

		for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
		{
			if (tID < stride)
			{
				sdata[tID] += sdata[tID + stride];
			}
			__syncthreads();
		}

		if (tID < 32){ warpReduce(sdata, tID); }

		if (tID == 0)
		{
			outdata[blockIdx.x] = sdata[0];
		}
	}
}

template <typename T>
void offset_momentum(int nbodies, planet<T> *bodies) {
	T px = 0.0, py = 0.0, pz = 0.0;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reduceSum<type>, 0, nbodies);
	gridSize = ((nbodies + blockSize - 1) / blockSize);

	if (gridSize > 1){
		gridSize = (int)ceil( (float)gridSize / 2);
	}
	else{
		blockSize = (int)ceil( (float)blockSize / 2);
	}

	T *d_reducedArray; cudaMalloc((void**)&d_reducedArray, gridSize * sizeof(T));
	T *h_reducedArray; h_reducedArray = new T[gridSize];
	planet<T> *d_bodies; cudaMalloc((void**)&d_bodies, nbodies * sizeof(planet<T>));

	cudaMemcpy(d_bodies, bodies, nbodies * sizeof(planet<T>), cudaMemcpyHostToDevice);
	reduceSum << <gridSize, blockSize, nbodies * sizeof(planet<T>) >> >(d_bodies, d_reducedArray, 1, nbodies);
	cudaMemcpy(h_reducedArray, d_reducedArray, gridSize * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 1; i < gridSize; i++){
		h_reducedArray[0] += h_reducedArray[i];
	}
	px = h_reducedArray[0];

	//cudaMemcpy(d_test, test, 1000 * sizeof(int), cudaMemcpyHostToDevice);
	reduceSum << <gridSize, blockSize, nbodies * sizeof(planet<T>) >> >(d_bodies, d_reducedArray, 2, nbodies);
	cudaMemcpy(h_reducedArray, d_reducedArray, gridSize * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 1; i < gridSize; i++){
		h_reducedArray[0] += h_reducedArray[i];
	}
	py = h_reducedArray[0];

	//cudaMemcpy(d_test, test, 1000 * sizeof(int), cudaMemcpyHostToDevice);
	reduceSum << <gridSize, blockSize, nbodies * sizeof(planet<T>) >> >(d_bodies, d_reducedArray, 3, nbodies);
	cudaMemcpy(h_reducedArray, d_reducedArray, gridSize * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 1; i < gridSize; i++){
		h_reducedArray[0] += h_reducedArray[i];
	}
	pz = h_reducedArray[0];

	bodies[0].vx = -px / solar_mass;
	bodies[0].vy = -py / solar_mass;
	bodies[0].vz = -pz / solar_mass;
}

struct planet<type> golden_bodies[5] = {
	{                               /* sun */
		0, 0, 0, 0, 0, 0, solar_mass
	},

	{                               /* jupiter */
		4.84143144246472090e+00,
		-1.16032004402742839e+00,
		-1.03622044471123109e-01,
		1.66007664274403694e-03 * days_per_year,
		7.69901118419740425e-03 * days_per_year,
		-6.90460016972063023e-05 * days_per_year,
		9.54791938424326609e-04 * solar_mass
	},

	{                               /* saturn */
		8.34336671824457987e+00,
		4.12479856412430479e+00,
		-4.03523417114321381e-01,
		-2.76742510726862411e-03 * days_per_year,
		4.99852801234917238e-03 * days_per_year,
		2.30417297573763929e-05 * days_per_year,
		2.85885980666130812e-04 * solar_mass
	},

	{                               /* uranus */
		1.28943695621391310e+01,
		-1.51111514016986312e+01,
		-2.23307578892655734e-01,
		2.96460137564761618e-03 * days_per_year,
		2.37847173959480950e-03 * days_per_year,
		-2.96589568540237556e-05 * days_per_year,
		4.36624404335156298e-05 * solar_mass
	},

	{                               /* neptune */
		1.53796971148509165e+01,
		-2.59193146099879641e+01,
		1.79258772950371181e-01,
		2.68067772490389322e-03 * days_per_year,
		1.62824170038242295e-03 * days_per_year,
		-9.51592254519715870e-05 * days_per_year,
		5.15138902046611451e-05 * solar_mass
	}
};

const type DT{ 1e-2 };
const type RECIP_DT{ 1.0 / DT };

/*
* Rescale certain properties of bodies. That allows doing
* consequential advance()'s as if dt were equal to 1.0.
*
* When all advances done, rescale bodies back to obtain correct energy.
*/
template <typename T>
__global__ void scale_bodies(int nBodies, planet<T> *bodies, T scale){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < nBodies){
		bodies[idx].mass *= scale * scale;
		bodies[idx].vx *= scale;
		bodies[idx].vy *= scale;
		bodies[idx].vz *= scale;
	}
}

template <typename T>
void init_random_bodies(int nbodies, planet<T> *bodies) {

	for (int i = 0; i < nbodies; ++i) {
		bodies[i].x = (T)rand() / RAND_MAX;
		bodies[i].y = (T)rand() / RAND_MAX;
		bodies[i].z = (T)rand() / RAND_MAX;
		bodies[i].vx = (T)rand() / RAND_MAX;
		bodies[i].vy = (T)rand() / RAND_MAX;
		bodies[i].vz = (T)rand() / RAND_MAX;
		bodies[i].mass = (T)rand() / RAND_MAX;
	}
}

int main(int argc, char ** argv) {

	//CPU
	int niters = 1000, nbodies = 5;
	if (argc > 1) { niters = atoi(argv[1]); }
	if (argc > 2) { nbodies = atoi(argv[2]); }

	std::cout << "niters=" << niters << " nbodies=" << nbodies << '\n';

	planet<type> *bodies;
	if (argc == 1) {
		bodies = golden_bodies; // Check accuracy with 1000 solar system iterations
	}
	else {
		bodies = new planet<type>[nbodies];
		init_random_bodies(nbodies, bodies);

		//planet<type> *cudabodies;
		//cudamalloc((void**)&cudabodies, nbodies * sizeof(planet<type>));
		//
		//cudaoccupancymaxpotentialblocksize(&mingridsize, &blocksize, init_random_bodies<type>, 0, nbodies);
		//gridsize = (nbodies + blocksize - 1) / blocksize;
		//init_random_bodies<<<gridsize, blocksize >>>(nbodies, cudabodies);

		//cudamemcpy(bodies, cudabodies, nbodies * sizeof(planet<type>), cudamemcpydevicetohost);
 	}

	auto t1 = std::chrono::steady_clock::now();
	offset_momentum(nbodies, bodies); //GPU
	type e1 = energy(nbodies, bodies); //GPU

	planet<type> *cudabodies;
	cudaMalloc((void**)&cudabodies, nbodies * sizeof(planet<type>));
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scale_bodies<type>, 0, nbodies);
	gridSize = (nbodies + blockSize - 1) / blockSize;
	cudaMemcpy(cudabodies, bodies, nbodies * sizeof(planet<type>), cudaMemcpyHostToDevice);
	scale_bodies<<<gridSize, blockSize>>>(nbodies, cudabodies, DT);
	cudaMemcpy(bodies, cudabodies, nbodies * sizeof(planet<type>), cudaMemcpyDeviceToHost);

	for (int i = 1; i <= niters; ++i)  {
		advance(nbodies, bodies);
	}

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scale_bodies<type>, 0, nbodies);
	gridSize = (nbodies + blockSize - 1) / blockSize;
	cudaMemcpy(cudabodies, bodies, nbodies * sizeof(planet<type>), cudaMemcpyHostToDevice);
	scale_bodies<<<gridSize, blockSize >>>(nbodies, cudabodies, RECIP_DT);
	cudaMemcpy(bodies, cudabodies, nbodies * sizeof(planet<type>), cudaMemcpyDeviceToHost);

	//scale_bodies(nbodies, bodies, RECIP_DT);

	type e2 = energy(nbodies, bodies);
	auto t2 = std::chrono::steady_clock::now();
	auto diff = t2 - t1;

	std::cout << std::setprecision(9);
	std::cout << e1 << '\n' << e2 << '\n';
	std::cout << std::fixed << std::setprecision(3);
	std::cout << std::chrono::duration<double>(diff).count() << " seconds.\n";

	if (argc != 1) { delete[] bodies; }
	return 0;
}
