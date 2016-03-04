///*
//Running without arguments is equivalent to 1000 iterations with the
//5 celestial objects declared in the golden_bodies array.
//
//$ nbody.exe 1000 5
//
//The output of this shows the energy before and after the simulation,
//and should be:
//
//-0.169075164
//-0.169087605
//*/
//
//#include <cuda_runtime.h>
//#include <cuda_occupancy.h>
//#include <malloc.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include <chrono>
//#include <cmath>
//#include <cstdlib>
//#include <iomanip>
//#include <iostream>
//#include <curand.h>
//#include <curand_kernel.h>
//#include <algorithm>
//
//using type = double;
//
//const type pi{ 3.141592653589793 };
//const type solar_mass{ 4 * pi * pi };
//const type days_per_year{ 365.24 };
//
//template <typename T>
//struct planet {
//	T x, y, z;
//	T vx, vy, vz;
//	T mass;
//};
//
//
//template <typename T>
//void advance(int nbodies, planet<T> *bodies)
//{
//	int i, j;
//
//	for (i = 0; i < nbodies; ++i) {
//		planet<T> &b = bodies[i];
//
//		for (j = i + 1; j < nbodies; j++) {
//			planet<T> &b2 = bodies[j];
//			T dx = b.x - b2.x;
//			T dy = b.y - b2.y;
//			T dz = b.z - b2.z;
//			T inv_distance = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
//			T mag = inv_distance * inv_distance * inv_distance;
//			b.vx -= dx * b2.mass * mag;
//			b.vy -= dy * b2.mass * mag;
//			b.vz -= dz * b2.mass * mag;
//			b2.vx += dx * b.mass  * mag;
//			b2.vy += dy * b.mass  * mag;
//			b2.vz += dz * b.mass  * mag;
//		}
//	}
//
//	for (i = 0; i < nbodies; ++i) {
//		planet<T> &b = bodies[i];
//		b.x += b.vx;
//		b.y += b.vy;
//		b.z += b.vz;
//	}
//}
//
////template<typename T>
////__global__ void advanceKernel(int nbodies, planet<T> *bodies){
////
////	unsigned int tID = threadIdx.x;
////	unsigned int i = tID + blockIdx.x * blockDim.x;
////
////	if (i < nbodies)
////	{
////		for (int iter = i + 1; iter < nbodies; iter++){
////			planet<T> &b = bodies[i];
////			planet<T> &b2 = bodies[iter];
////			T dx = b.x - b2.x;
////			T dy = b.y - b2.y;
////			T dz = b.z - b2.z;
////			T inv_distance = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
////			T mag = inv_distance * inv_distance * inv_distance;
////			b.vx -= dx * b2.mass * mag;
////			b.vy -= dy * b2.mass * mag;
////			b.vz -= dz * b2.mass * mag;
////			b2.vx += dx * b.mass  * mag;
////			b2.vy += dy * b.mass  * mag;
////			b2.vz += dz * b.mass  * mag;
////
////		}
////	}
////
////	__syncthreads();
////
////	if (i < nbodies){
////		planet<T> &b = bodies[i];
////		b.x += b.vx;
////		b.y += b.vy;
////		b.z += b.vz;
////	}
////
////}
//
//template <typename T>
//__global__ void advanceVelocities(int nbodies, planet<T> *bodies)
//{
//	int i = threadIdx.x + blockIdx.x*blockDim.x;
//
//	if (i < nbodies)
//	{
//		planet<T> &b = bodies[i];
//		for (int j = i + 1; j < nbodies; ++j)
//		{
//			planet<T> &b2 = bodies[j];
//			T dx = b.x - b2.x;
//			T dy = b.y - b2.y;
//			T dz = b.z - b2.z;
//			T inv_distance = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
//			T mag = inv_distance * inv_distance * inv_distance;
//			b.vx -= dx * b2.mass * mag;
//			b.vy -= dy * b2.mass * mag;
//			b.vz -= dz * b2.mass * mag;
//			b2.vx += dx * b.mass  * mag;
//			b2.vy += dy * b.mass  * mag;
//			b2.vz += dz * b.mass  * mag;
//		}
//	}
//}
//
//template <typename T>
//__global__ void advancePositions(int nbodies, planet<T> *bodies)
//{
//	int i = threadIdx.x + blockIdx.x*blockDim.x;
//
//	if (i < nbodies)
//	{
//		planet<T> &b = bodies[i];
//		b.x += b.vx;
//		b.y += b.vy;
//		b.z += b.vz;
//	}
//}
//
//template <typename T>
//void advance_gpued(int nbodies, planet<T> *bodies)
//{
//	advanceVelocities << <1, nbodies >> >(nbodies, bodies);
//
//	advancePositions << <1, nbodies >> >(nbodies, bodies);
//}
//
//template <typename T>
//T energy(int nbodies, planet<T> *bodies) {
//	T e = 0.0;
//	T e2 = 0.0;
//
//	//GPU
//	for (int i = 0; i < nbodies; ++i) {
//		planet<T> &b = bodies[i];
//		e += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
//
//		for (int j = i + 1; j < nbodies; j++) {
//			planet<T> &b2 = bodies[j];
//			T dx = b.x - b2.x;
//			T dy = b.y - b2.y;
//			T dz = b.z - b2.z;
//			T distance = sqrt(dx * dx + dy * dy + dz * dz);
//			e -= (b.mass * b2.mass) / distance;
//		}
//	}
//
//	return e;
//}
//
//template <typename T>
//void offset_momentum(int nbodies, planet<T> *bodies) {
//	T px = 0.0, py = 0.0, pz = 0.0;
//
//	//GPU
//	for (int i = 0; i < nbodies; ++i) {
//		px += bodies[i].vx * bodies[i].mass;
//		py += bodies[i].vy * bodies[i].mass;
//		pz += bodies[i].vz * bodies[i].mass;
//	}
//
//	bodies[0].vx = -px / solar_mass;
//	bodies[0].vy = -py / solar_mass;
//	bodies[0].vz = -pz / solar_mass;
//}
//
//struct planet<type> golden_bodies[5] = {
//	{                               /* sun */
//		0, 0, 0, 0, 0, 0, solar_mass
//	},
//
//	{                               /* jupiter */
//		4.84143144246472090e+00,
//		-1.16032004402742839e+00,
//		-1.03622044471123109e-01,
//		1.66007664274403694e-03 * days_per_year,
//		7.69901118419740425e-03 * days_per_year,
//		-6.90460016972063023e-05 * days_per_year,
//		9.54791938424326609e-04 * solar_mass
//	},
//
//	{                               /* saturn */
//		8.34336671824457987e+00,
//		4.12479856412430479e+00,
//		-4.03523417114321381e-01,
//		-2.76742510726862411e-03 * days_per_year,
//		4.99852801234917238e-03 * days_per_year,
//		2.30417297573763929e-05 * days_per_year,
//		2.85885980666130812e-04 * solar_mass
//	},
//
//	{                               /* uranus */
//		1.28943695621391310e+01,
//		-1.51111514016986312e+01,
//		-2.23307578892655734e-01,
//		2.96460137564761618e-03 * days_per_year,
//		2.37847173959480950e-03 * days_per_year,
//		-2.96589568540237556e-05 * days_per_year,
//		4.36624404335156298e-05 * solar_mass
//	},
//
//	{                               /* neptune */
//		1.53796971148509165e+01,
//		-2.59193146099879641e+01,
//		1.79258772950371181e-01,
//		2.68067772490389322e-03 * days_per_year,
//		1.62824170038242295e-03 * days_per_year,
//		-9.51592254519715870e-05 * days_per_year,
//		5.15138902046611451e-05 * solar_mass
//	}
//};
//
//const type DT{ 1e-2 };
//const type RECIP_DT{ 1.0 / DT };
//
///*
//* Rescale certain properties of bodies. That allows doing
//* consequential advance()'s as if dt were equal to 1.0.
//*
//* When all advances done, rescale bodies back to obtain correct energy.
//*/
//template <typename T>
//void scale_bodies(int nbodies, planet<T> *bodies, T scale) {
//	//GPU
//	for (int i = 0; i < nbodies; ++i) {
//		bodies[i].mass *= scale*scale;
//		bodies[i].vx *= scale;
//		bodies[i].vy *= scale;
//		bodies[i].vz *= scale;
//	}
//}
//
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
//
//int main(int argc, char ** argv) {
//
//	//CPU
//	int niters = 1000, nbodies = 5;
//	if (argc > 1) { niters = atoi(argv[1]); }
//	if (argc > 2) { nbodies = atoi(argv[2]); }
//
//	std::cout << "niters=" << niters << " nbodies=" << nbodies << '\n';
//
//	planet<type> *bodies;
//	if (argc == 1) {
//		bodies = golden_bodies; // Check accuracy with 1000 solar system iterations
//	}
//	else {
//		bodies = new planet<type>[nbodies];
//		init_random_bodies(nbodies, bodies); //GPU
//	}
//
//	auto offsetStart = std::chrono::steady_clock::now();
//	offset_momentum(nbodies, bodies); //GPU
//	auto offsetStop = std::chrono::steady_clock::now();
//
//	auto energyStart = std::chrono::steady_clock::now();
//	type e1 = energy(nbodies, bodies); //GPU
//	auto energyStop = std::chrono::steady_clock::now();
//
//	auto scaleStart = std::chrono::steady_clock::now();
//	scale_bodies(nbodies, bodies, DT);
//	auto scaleStop = std::chrono::steady_clock::now();
//
//	auto advanceStart = std::chrono::steady_clock::now();
//
//	//planet<type> *devBodies = nullptr;
//	//cudaMalloc(&devBodies, nbodies*sizeof(planet<type>));
//	//cudaMemcpy(devBodies, bodies, nbodies*sizeof(planet<type>), cudaMemcpyHostToDevice);
//
//	for (int i = 1; i <= niters; ++i)  {
//		//advanceKernel<<<1, nbodies>>>(nbodies, devBodies);
//		//advance_gpued(nbodies, devBodies);
//		advance(nbodies, bodies);
//	}
//
//	//cudaMemcpy(bodies, devBodies, nbodies*sizeof(planet<type>), cudaMemcpyDeviceToHost);
//
//	auto advanceStop = std::chrono::steady_clock::now();
//
//	auto scale2Start = std::chrono::steady_clock::now();
//	scale_bodies(nbodies, bodies, RECIP_DT);
//	auto scale2Stop = std::chrono::steady_clock::now();
//
//	auto energy2Start = std::chrono::steady_clock::now();
//	type e2 = energy(nbodies, bodies);
//	auto energy2Stop = std::chrono::steady_clock::now();
//
//	auto offsetDiff = offsetStop - offsetStart;
//	auto energyDiff = energyStop - energyStart;
//	auto scaleDiff = scaleStop - scaleStart;
//	auto advanceDiff = advanceStop - advanceStart;
//	auto scale2Diff = scale2Stop - scale2Start;
//	auto energy2Diff = energy2Stop - energy2Start;
//
//	std::cout << e1 << '\n' << e2 << '\n';
//	std::cout << std::fixed << std::setprecision(3);
//	std::cout << std::chrono::duration<double>(offsetDiff + energyDiff + scaleDiff + advanceDiff + scale2Diff + energy2Diff).count() << " seconds.\n" << std::endl;
//	std::cout << "offsetTime: " << std::chrono::duration<double>(offsetDiff).count() << std::endl;
//	std::cout << "energyTime: " << std::chrono::duration<double>(energyDiff).count() << std::endl;
//	std::cout << "scaleTime: " << std::chrono::duration<double>(scaleDiff).count() << std::endl;
//	std::cout << "advanceTime: " << std::chrono::duration<double>(advanceDiff).count() << std::endl;
//	std::cout << "scale2Time: " << std::chrono::duration<double>(scale2Diff).count() << std::endl;
//	std::cout << "energy2Time: " << std::chrono::duration<double>(energy2Diff).count() << std::endl << std::endl;
//
//	if (argc != 1) { delete[] bodies; }
//	return 0;
//}
