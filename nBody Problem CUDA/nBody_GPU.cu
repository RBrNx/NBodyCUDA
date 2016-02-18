//GPU code goes here
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//__global__ void initValues(int *input, int *output, int size){
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//	if (idx < size) { output[idx] = input[idx] * 2; }
//}
//
//int main(){
//	const int size = 1000000;
//	srand(time(NULL));
//
//	int blockSize;
//	int minGridSize;
//	int gridSize;
//
//	int* h_Array = (int*) malloc(size * sizeof(int));
//	int* h_testArray = (int*)malloc(size * sizeof(int));
//
//	int* d_InputArray; cudaMalloc((void**)&d_InputArray, size * sizeof(int));
//	int* d_OutputArray; cudaMalloc((void**)&d_OutputArray, size * sizeof(int));
//
//	//Test
//	for (int i = 0; i < size; i++){
//		h_Array[i] = i;
//		h_testArray[i] = h_Array[i] * 2;
//	}
//
//	cudaMemcpy(d_InputArray, h_Array, size * sizeof(int), cudaMemcpyHostToDevice);
//
//	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initValues, 0, size);
//
//	gridSize = (size + blockSize - 1) / blockSize;
//
//	initValues <<<gridSize, blockSize >>>(d_InputArray, d_OutputArray, size);
//
//	cudaMemcpy(h_Array, d_OutputArray, size*sizeof(int), cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < size; i++){
//		if (h_Array[i] != h_testArray[i]){
//			printf("Error at %i ! Host = %i, Device = %i \n", i, h_testArray[i], h_Array[i]);
//		}
//	}
//
//	int random = rand() % size;
//	printf("Random Number: %i, Host Value at %i, Device Value at %i \n", random, h_testArray[random], h_Array[random]);
//
//	printf("Test Passed \n");
//
//}

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

template <typename T>
T energy(int nbodies, planet<T> *bodies) {
	T e = 0.0;

	//GPU
	for (int i = 0; i < nbodies; ++i) {
		planet<T> &b = bodies[i];
		e += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);

		for (int j = i + 1; j < nbodies; j++) {
			planet<T> &b2 = bodies[j];
			T dx = b.x - b2.x;
			T dy = b.y - b2.y;
			T dz = b.z - b2.z;
			T distance = sqrt(dx * dx + dy * dy + dz * dz);
			e -= (b.mass * b2.mass) / distance;
		}
	}
	return e;
}

template <typename T>
void offset_momentum(int nbodies, planet<T> *bodies) {
	T px = 0.0, py = 0.0, pz = 0.0;

	//GPU
	for (int i = 0; i < nbodies; ++i) {
		px += bodies[i].vx * bodies[i].mass;
		py += bodies[i].vy * bodies[i].mass;
		pz += bodies[i].vz * bodies[i].mass;
	}

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
void scale_bodies(int nbodies, planet<T> *bodies, T scale) {
	//GPU
	for (int i = 0; i < nbodies; ++i) {
		bodies[i].mass *= scale*scale;
		bodies[i].vx *= scale;
		bodies[i].vy *= scale;
		bodies[i].vz *= scale;
	}
}

//template <typename T>
//void init_random_bodies(int nbodies, planet<T> *bodies) {
//
//	//GPU
//	for (int i = 0; i < nbodies; ++i) {
//		bodies[i].x = (T)rand() / RAND_MAX;
//		bodies[i].y = (T)rand() / RAND_MAX;
//		bodies[i].z = (T)rand() / RAND_MAX;
//		bodies[i].vx = (T)rand() / RAND_MAX;
//		bodies[i].vy = (T)rand() / RAND_MAX;
//		bodies[i].vz = (T)rand() / RAND_MAX;
//		bodies[i].mass = (T)rand() / RAND_MAX;
//	}
//}

template <typename T>
__global__ void init_random_bodies(int nBodies, planet<T> *bodies) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	curandState state;
	curand_init((unsigned long long)clock() + idx, 0, 0, &state);

	if (idx < nBodies){
		bodies[idx].x = curand_uniform_double(&state) + 0.0001;
		bodies[idx].y = curand_uniform_double(&state) + 0.0001;
		bodies[idx].z = curand_uniform_double(&state) + 0.0001;
		bodies[idx].vx = curand_uniform_double(&state) + 0.0001;
		bodies[idx].vy = curand_uniform_double(&state) + 0.0001;
		bodies[idx].vz = curand_uniform_double(&state) + 0.0001;
		bodies[idx].mass = curand_uniform_double(&state) + 0.0001;
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

		planet<type> *cudaBodies;
		cudaMalloc((void**)&cudaBodies, nbodies * sizeof(planet<type>));

		cudaMemcpy(cudaBodies, bodies, nbodies * sizeof(planet<type>), cudaMemcpyHostToDevice);
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_random_bodies<type>, 0, nbodies);
		gridSize = (nbodies + blockSize - 1) / blockSize;
		init_random_bodies<<<gridSize, blockSize >>>(nbodies, cudaBodies);

		cudaMemcpy(bodies, cudaBodies, nbodies * sizeof(planet<type>), cudaMemcpyDeviceToHost);

		//init_random_bodies(nbodies, bodies); //Old Function
 	}

	auto t1 = std::chrono::steady_clock::now();
	offset_momentum(nbodies, bodies); //GPU
	type e1 = energy(nbodies, bodies); //GPU
	scale_bodies(nbodies, bodies, DT);

	for (int i = 1; i <= niters; ++i)  {
		advance(nbodies, bodies);
	}
	scale_bodies(nbodies, bodies, RECIP_DT);

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
