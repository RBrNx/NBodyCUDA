/*
Running without arguments is equivalent to 1000 iterations with the
5 celestial objects declared in the golden_bodies array.

$ nbody.exe 1000 5

The output of this shows the energy before and after the simulation,
and should be:

-0.169075164
-0.169087605
*/

#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <algorithm>

/*
	inline function that will print any CUDA error codes, including the line number and file name where the error is present
*/
#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using type = double;

const type pi{ 3.141592653589793 };
const type solar_mass{ 4 * pi * pi };
const type days_per_year{ 365.24 };

const type DT{ 1e-2 };
const type RECIP_DT{ 1.0 / DT };

int blockSize;
int minGridSize;
int gridSize;

template <typename T>
struct planet {
	T x, y, z;
	T vx, vy, vz;
	T mass;
};

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

/*
	advanceKernel replaces the advace() function from the Serial Code.
*/
template<typename T>
__global__ void advanceKernel(int nbodies, planet<T> *bodies){

	unsigned int tID = threadIdx.x;
	unsigned int i = tID + blockIdx.x * blockDim.x;

	if (i < nbodies)
	{
		for (int iter = i + 1; iter < nbodies; iter++){
			planet<T> &b = bodies[i];
			planet<T> &b2 = bodies[iter];
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

	__syncthreads();

	if (i < nbodies){
		planet<T> &b = bodies[i];
		b.x += b.vx;
		b.y += b.vy;
		b.z += b.vz;
	}

}

/*
	energyKernel replaces the energy() function from the Serial Code.
	Split into two 'loops'. First loop calculates all the additions and reduces them.
	Second loop below the line, calculates all the subtractions and reduces them.
*/
template <typename T>
__global__ void energyKernel(int nbodies, T *addReduc, T *subReduc, planet<T> *bodies){
	extern __shared__ T e[];

	unsigned int tID = threadIdx.x;
	unsigned int i = tID + blockIdx.x * (blockDim.x * 2);

	if (i < nbodies){
		planet<T> &b = bodies[i];
		e[tID] = 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
	}

	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tID < stride)
		{
			e[tID] += e[tID + stride];
		}
		__syncthreads();
	}

	if (tID == 0)
	{
		addReduc[blockIdx.x] = e[0];
	}

	__syncthreads();

	//------------------------------------------------

	e[tID] = 0;

	if (i < nbodies)
	{
		for (int iter = i + 1; iter < nbodies; iter++){
			planet<T> &b = bodies[i];
			planet<T> &b2 = bodies[iter];
			T dx = b.x - b2.x;
			T dy = b.y - b2.y;
			T dz = b.z - b2.z;
			T distance = sqrt(dx * dx + dy * dy + dz * dz);
			T var = ((b.mass * b2.mass) / distance);
			e[tID] += var;
		}
	}

	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tID < stride)
		{
			e[tID] += e[tID + stride];
		}
		__syncthreads();
	}

	if (tID == 0)
	{
		subReduc[blockIdx.x] = e[0];
	}
}

/*
	energy function now just sets up some host/device arrays and launches the energyKernel to calculate the energy values.
	Additions are stored in h_addArray, Subtractions are stored in h_subArray. 
	h_addArray - h_subArray calculates the additions minus the subtractions all in one go.
*/
template <typename T>
T energy(int nbodies, planet<T> *bodies)
{
	T *h_addArray = new T[gridSize];
	T *h_subArray = new T[gridSize];

	T *d_addArray; cudaMalloc((void**)&d_addArray, gridSize * sizeof(T));
	T *d_subArray; cudaMalloc((void**)&d_subArray, gridSize * sizeof(T));

	energyKernel<<<gridSize, blockSize, nbodies * sizeof(T)>>>(nbodies, d_addArray, d_subArray, bodies);
	gpuErr(cudaPeekAtLastError());
	gpuErr(cudaMemcpy(h_addArray, d_addArray, gridSize * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErr(cudaMemcpy(h_subArray, d_subArray, gridSize * sizeof(T), cudaMemcpyDeviceToHost));

	for (int i = 1; i < gridSize; i++){
		h_addArray[0] += h_addArray[i];
		h_subArray[0] += h_subArray[i];
	}

	T e = h_addArray[0] - h_subArray[0];

	return e;
}

/*
	offsetKernel stores the local variables px, py and pz, and calls offsetReduction to calculate these.
*/
template <typename T>
__global__ void offsetKernel(planet<T> *bodies, T *outdata, int nbodies, int gridSize, T solar_mass){
	T px = 0.0, py = 0.0, pz = 0.0;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	offsetReduction(bodies, outdata, 1, nbodies);
	if (i < gridSize && i > 0){
		outdata[0] += outdata[i];
	}
	px = outdata[0];
	__syncthreads();

	offsetReduction(bodies, outdata, 2, nbodies);
	if (i < gridSize && i > 0){
		outdata[0] += outdata[i];
	}
	py = outdata[0];
	__syncthreads();

	offsetReduction(bodies, outdata, 3, nbodies);
	if (i < gridSize && i > 0){
		outdata[0] += outdata[i];
	}
	pz = outdata[0];
	__syncthreads();

	bodies[0].vx = -px / solar_mass;
	bodies[0].vy = -py / solar_mass;
	bodies[0].vz = -pz / solar_mass;

}

/*
	offsetReduction replaces most of the offset_mometum() function from the Serial Code.
	arrayIdent == 1 calculates px, 2 calculates py and 3 calculates pz.
*/
template <typename T>
__device__ void offsetReduction(planet<T> *bodies, T *outdata, int arrayIdent, int nbodies){
	extern __shared__ T sdata[];

	unsigned int tID = threadIdx.x;
	unsigned int i = tID + blockIdx.x * blockDim.x;

	if (arrayIdent == 1){
		if (i < nbodies){
			sdata[tID] = bodies[i].vx * bodies[i].mass;
		}

		__syncthreads();
	}

	if (arrayIdent == 2){
		if (i < nbodies){
			sdata[tID] = (bodies[i].vy * bodies[i].mass);
		}
		__syncthreads();
	}

	if (arrayIdent == 3){
		if (i < nbodies){
			sdata[tID] = (bodies[i].vz * bodies[i].mass);
		}
		__syncthreads();
	}

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>=1)
	{
		if (tID < stride)
		{
			sdata[tID] += sdata[tID + stride];
		}
		__syncthreads();
	}

	if (tID == 0)
	{
		outdata[blockIdx.x] = sdata[0];
	}
}

/*
	offset_mometum allocates space for one variable on the GPU and launches the offsetKernel.
	This variable is used to store px, py and pz after each of the offsetReduction calls.
*/
template <typename T>
void offset_momentum(int nbodies, planet<T> *bodies) {
	T *d_reducedArray; cudaMalloc((void**)&d_reducedArray, gridSize * sizeof(T));

	offsetKernel<<<gridSize, blockSize, nbodies * sizeof(T)>>>(bodies, d_reducedArray, nbodies, gridSize, solar_mass);
	gpuErr(cudaPeekAtLastError());

	cudaFree(d_reducedArray);
}

/*
	scale_bodiesKernel replaces scale_bodies() from the Serial Code.
*/
template <typename T>
__global__ void scale_bodiesKernel(int nBodies, planet<T> *bodies, T scale){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < nBodies){
		bodies[idx].mass *= scale * scale;
		bodies[idx].vx *= scale;
		bodies[idx].vy *= scale;
		bodies[idx].vz *= scale;
	}
}
/*
	init_random_bodies() is exactly the same as the Serial Code version, this has not been parallelized.
*/
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

/*
	isPowerOfTwo() returns true if x is a power of 2, and false if otherwise.
	Used to round our number of threads up to the nearest power of two if it is not a power of two.
*/
int isPowerOfTwo(unsigned int x)
{
	return ((x != 0) && !(x & (x - 1)));
}

int main(int argc, char ** argv) {
	int niters = 1000, nbodies = 5;
	if (argc > 1) { niters = atoi(argv[1]); }
	if (argc > 2) { nbodies = atoi(argv[2]); }

	std::cout << "niters=" << niters << " nbodies=" << nbodies << '\n';

	planet<type> *bodies;
	planet<type> *cudabodies;
	if (argc == 1) {
		bodies = golden_bodies;
	}
	else {
		bodies = new planet<type>[nbodies];
		init_random_bodies(nbodies, bodies);
	}

	gpuErr(cudaMalloc((void**)&cudabodies, nbodies * sizeof(planet<type>))); //Allocate memory on GPU for bodies
	gpuErr(cudaMemcpy(cudabodies, bodies, nbodies * sizeof(planet<type>), cudaMemcpyHostToDevice)); //Copy host bodies into allocated memory
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scale_bodiesKernel<type>, 0, nbodies); //Returns a suggested gridsize and blocksize based on the number of bodies
	gridSize = (nbodies + blockSize - 1) / blockSize;

	//Rounds blocksize to a power of two. Reduction algorithms will not work without a power of two.
	if (!isPowerOfTwo(blockSize)){ 
		blockSize = pow(2, ceil(log(blockSize) / log(2)));
	}
	if (blockSize > 1024){
		int multiple = blockSize / 1024;
		gridSize = multiple;
	}

	//Offset Momemtum
	auto offsetStart = std::chrono::steady_clock::now();
	offset_momentum(nbodies, cudabodies);
	cudaThreadSynchronize();
	auto offsetStop = std::chrono::steady_clock::now();

	//Starting Energy
	auto energyStart = std::chrono::steady_clock::now();
	type e1 = energy(nbodies, cudabodies);
	cudaThreadSynchronize();
	auto energyStop = std::chrono::steady_clock::now();

	//Scale Bodies down
	auto scaleStart = std::chrono::steady_clock::now();
	scale_bodiesKernel<<<gridSize, blockSize>>>(nbodies, cudabodies, DT);
	gpuErr(cudaPeekAtLastError());
	cudaThreadSynchronize();
	auto scaleStop = std::chrono::steady_clock::now();

	//Advance for niters iterations
	auto advanceStart = std::chrono::steady_clock::now();
	for (int i = 1; i <= niters; ++i)  {
		advanceKernel<<<gridSize, blockSize>>>(nbodies, cudabodies);
		gpuErr(cudaPeekAtLastError());
		cudaThreadSynchronize();
	}
	auto advanceStop = std::chrono::steady_clock::now();

	//Scale Bodies up
	auto scale2Start = std::chrono::steady_clock::now();
	scale_bodiesKernel<<<gridSize, blockSize>>>(nbodies, cudabodies, RECIP_DT);
	gpuErr(cudaPeekAtLastError());
	cudaThreadSynchronize();
	auto scale2Stop = std::chrono::steady_clock::now();

	//Finished Energy
	auto energy2Start = std::chrono::steady_clock::now();
	type e2 = energy(nbodies, cudabodies);
	cudaThreadSynchronize();
	auto energy2Stop = std::chrono::steady_clock::now();

	//Calculate timings and print to console
	auto offsetDiff = offsetStop - offsetStart;
	auto energyDiff = energyStop - energyStart;
	auto scaleDiff = scaleStop - scaleStart;
	auto advanceDiff = advanceStop - advanceStart;
	auto scale2Diff = scale2Stop - scale2Start;
	auto energy2Diff = energy2Stop - energy2Start;

	std::cout << e1 << '\n' << e2 << '\n';
	std::cout << std::fixed << std::setprecision(3);
	std::cout << std::chrono::duration<double>(offsetDiff + energyDiff + scaleDiff + advanceDiff + scale2Diff + energy2Diff).count() << " seconds.\n" << std::endl;
	std::cout << "offsetTime: " << std::chrono::duration<double>(offsetDiff).count() << std::endl;
	std::cout << "energyTime: " << std::chrono::duration<double>(energyDiff).count() << std::endl;
	std::cout << "scaleTime: " << std::chrono::duration<double>(scaleDiff).count() << std::endl;
	std::cout << "advanceTime: " << std::chrono::duration<double>(advanceDiff).count() << std::endl;
	std::cout << "scale2Time: " << std::chrono::duration<double>(scale2Diff).count() << std::endl;
	std::cout << "energy2Time: " << std::chrono::duration<double>(energy2Diff).count() << std::endl << std::endl;


	if (argc != 1) { delete[] bodies; }
	return 0;
}
