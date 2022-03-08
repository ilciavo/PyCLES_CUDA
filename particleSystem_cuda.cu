#include <helper_cuda.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
//#include <GL/glew.h>
//#include <cuda_gl_interop.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel.cu"
//#include "ParticleSystem.cuh"

#include <thrust/random.h>

extern "C"
{
    __global__ void initRandVectors(float *vectorX, float *vectorY, float *vectorZ, 
                                    float minX, float maxX,
                                    float minY, float maxY,
                                    float minZ, float maxZ,
                                    int numFloats)
    {
        int tid = blockIdx.x*gridDim.x + threadIdx.x;
        if (tid < numFloats){
            thrust::default_random_engine rng(tid);
            rng.discard(numFloats*tid);
            thrust::uniform_real_distribution<float> randPosX(minX, maxX);
            thrust::uniform_real_distribution<float> randPosY(minY, maxY);
            thrust::uniform_real_distribution<float> randPosZ(minZ, maxZ);
            vectorX[tid]=randPosX(rng);
            vectorY[tid]=randPosY(rng);
            vectorZ[tid]=randPosZ(rng);
        }
    }
    __global__ void interpolation(float *partPosX, float *partPosY, float *partPosZ,
                                  float *xi, float *yi, float *zi,
                                  float *vertxVelu, float *vertxVelv, float *vertxVelw,
                                  float *partVelu, float *partVelv, float *partVelw,
                                  float ximin, float ximax,
                                  float yimin, float yimax,
                                  float zimin, float zimax,
                                  float Nx, float Ny, float Nz,
                                  int gw, float deltaX, float deltaY, float deltaZ,                                  
                                  int numberParticles)

    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;

        if (tid < numberParticles){
            //Position of the particle
            float posX = partPosX[tid];
            float posY = partPosY[tid];
            float posZ = partPosZ[tid];

            //Index
            int idXL = int(floor((posX-ximin)/deltaX)+gw-1);  
            int idYL = int(floor((posY-yimin)/deltaY)+gw-1);
            int idZL = int(floor((posZ-zimin)/deltaZ)+gw-1);
            int idXR = idXL+1;  
            int idYR = idYL+1;
            int idZR = idZL+1;

            int nx = Nx+2*gw;
            int ny = Ny+2*gw;
            int nz = Nz+2*gw;

            //int id = idX*ly*lz+idY*lz+idZ
            int idA = (idXL)*ny*nz+(idYL)*nz+(idZL); 
            int idB = (idXR)*ny*nz+(idYL)*nz+(idZL);
            int idC = (idXR)*ny*nz+(idYR)*nz+(idZL);
            int idD = (idXL)*ny*nz+(idYR)*nz+(idZL);
            int idE = (idXL)*ny*nz+(idYL)*nz+(idZR); 
            int idF = (idXR)*ny*nz+(idYL)*nz+(idZR);
            int idG = (idXR)*ny*nz+(idYR)*nz+(idZR);
            int idH = (idXL)*ny*nz+(idYR)*nz+(idZR);

            //Velocities of vertices
            float velAu= vertxVelu[idA];
            float velAv= vertxVelv[idA];
            float velAw= vertxVelw[idA];

            float velBu= vertxVelu[idB];
            float velBv= vertxVelv[idB];
            float velBw= vertxVelw[idB];
            
            float velCu= vertxVelu[idC];
            float velCv= vertxVelv[idC];
            float velCw= vertxVelw[idC];

            float velDu= vertxVelu[idD];
            float velDv= vertxVelv[idD];
            float velDw= vertxVelw[idD];
            
            float velEu= vertxVelu[idE];
            float velEv= vertxVelv[idE];
            float velEw= vertxVelw[idE];

            float velFu= vertxVelu[idF];
            float velFv= vertxVelv[idF];
            float velFw= vertxVelw[idF];

            float velGu= vertxVelu[idG];
            float velGv= vertxVelv[idG];
            float velGw= vertxVelw[idG];

            float velHu= vertxVelu[idH];
            float velHv= vertxVelv[idH];
            float velHw= vertxVelw[idH];

            //Relative position of a particle with respect to its cell
            float dx = (posX - xi[idXL])/(deltaX);
            float dy = (posY - yi[idYL])/(deltaY);
            float dz = (posZ - zi[idZL])/(deltaZ);

            //Interpolate velocity 
            float velU, velV, velW;            
            velU = (1-dx)*(1-dy)*(1-dz)*velAu +
                    dx*(1-dy)*(1-dz)*velBu + 
                    dx*dy*(1-dz)*velCu +
                    (1-dx)*dy*(1-dz)*velDu +
                    (1-dx)*(1-dy)*dz*velEu + 
                    dx*(1-dy)*dz*velFu + 
                    dx*dy*dz*velGu +
                    (1-dx)*dy*dz*velHu;

            velV = (1-dx)*(1-dy)*(1-dz)*velAv +
                    dx*(1-dy)*(1-dz)*velBv + 
                    dx*dy*(1-dz)*velCv +
                    (1-dx)*dy*(1-dz)*velDv +
                    (1-dx)*(1-dy)*dz*velEv + 
                    dx*(1-dy)*dz*velFv + 
                    dx*dy*dz*velGv +
                    (1-dx)*dy*dz*velHv;

            velW = (1-dx)*(1-dy)*(1-dz)*velAw +
                    dx*(1-dy)*(1-dz)*velBw + 
                    dx*dy*(1-dz)*velCw +
                    (1-dx)*dy*(1-dz)*velDw +
                    (1-dx)*(1-dy)*dz*velEw + 
                    dx*(1-dy)*dz*velFw + 
                    dx*dy*dz*velGw +
                    (1-dx)*dy*dz*velHw;

            //Store in memory
            partVelu[tid] = velU;
            partVelv[tid] = velV;
            partVelw[tid] = velW;

        }
    }

        __global__ void interpolation2(float *partPosX, float *partPosY, float *partPosZ,
                                  float *xi, float *yi, float *zi,
                                  float *vertxVelu, float *vertxVelv, float *vertxVelw,
                                  float *partVelu, float *partVelv, float *partVelw,
                                  float ximin, float ximax,
                                  float yimin, float yimax,
                                  float zimin, float zimax,
                                  float Nx, float Ny, float Nz,
                                  int gw, float deltaX, float deltaY, float deltaZ,                                  
                                  int numberParticles)

    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;

        if (tid < numberParticles){
            //Position of the particle
            float posX = partPosX[tid];
            float posY = partPosY[tid];
            float posZ = partPosZ[tid];

            //Index
            int idXL = floor((posX-ximin)/deltaX)+gw-1;  
            int idYL = floor((posY-yimin)/deltaY)+gw-1;
            int idZL = floor((posZ-zimin)/deltaZ)+gw-1;
            int idXR = idXL+1;  
            int idYR = idYL+1;
            int idZR = idZL+1;

            int nx = Nx+2*gw;
            int ny = Ny+2*gw;
            int nz = Nz+2*gw;

            //int id = idX*ly*lz+idY*lz+idZ;
            int idA = (idXL)*ny*nz+(idYL)*nz+(idZL); 
            int idB = (idXR)*ny*nz+(idYL)*nz+(idZL);
            int idC = (idXR)*ny*nz+(idYR)*nz+(idZL);
            int idD = (idXL)*ny*nz+(idYR)*nz+(idZL);
            int idE = (idXL)*ny*nz+(idYL)*nz+(idZR); 
            int idF = (idXR)*ny*nz+(idYL)*nz+(idZR);
            int idG = (idXR)*ny*nz+(idYR)*nz+(idZR);
            int idH = (idXL)*ny*nz+(idYR)*nz+(idZR);

            //Velocities of vertices
            float velAu= vertxVelu[idA];
            float velAv= vertxVelv[idA];
            float velAw= vertxVelw[idA];

            float velBu= vertxVelu[idB];
            float velBv= vertxVelv[idB];
            float velBw= vertxVelw[idB];
            
            float velCu= vertxVelu[idC];
            float velCv= vertxVelv[idC];
            float velCw= vertxVelw[idC];

            float velDu= vertxVelu[idD];
            float velDv= vertxVelv[idD];
            float velDw= vertxVelw[idD];
            
            float velEu= vertxVelu[idE];
            float velEv= vertxVelv[idE];
            float velEw= vertxVelw[idE];

            float velFu= vertxVelu[idF];
            float velFv= vertxVelv[idF];
            float velFw= vertxVelw[idF];

            float velGu= vertxVelu[idG];
            float velGv= vertxVelv[idG];
            float velGw= vertxVelw[idG];

            float velHu= vertxVelu[idH];
            float velHv= vertxVelv[idH];
            float velHw= vertxVelw[idH];

            //Interpolate velocity 
            float velU, velV, velW;            
            velU = (velAu + velBu + velCu + velDu + velEu + velFu +velGu + velHu)/8.0;

            velV = (velAv + velBv + velCv + velDv + velEv + velFv +velGv + velHv)/8.0;

            velW = (velAw + velBw + velCw + velDw + velEw + velFw + velGw + velHw)/8.0;

            //Store in memory
            //partVel[tid] = vel;
            partVelu[tid] = velU;
            partVelv[tid] = velV;
            partVelw[tid] = velW;
        }
    }



    __global__ void interpolation4(float4 *partPos, float4 *vertxVel, 
                                  float4 *partVel, int numberFloats)
    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;

        if (tid < numberFloats){
            //Position of the particle
            float4 pos = partPos[tid];
            float dx = pos.x;
            float dy = pos.y;
            float dz = pos.z;


            //Velocities of vertices
            float4 velA = vertxVel[0];
            float4 velB = vertxVel[1];
            float4 velC = vertxVel[2];
            float4 velD = vertxVel[3];
            float4 velE = vertxVel[4];
            float4 velF = vertxVel[5];
            float4 velG = vertxVel[6];
            float4 velH = vertxVel[7];
            
            //Interpolate velocity 
            float4 vel;

            vel.x = (1-dx)*(1-dy)*(1-dz)*velA.x +
                    dx*(1-dy)*(1-dz)*velB.x + 
                    dx*dy*(1-dz)*velC.x +
                    (1-dx)*dy*(1-dz)*velD.x +
                    (1-dx)*(1-dy)*dz*velE.x + 
                    dx*(1-dy)*dz*velF.x + 
                    dx*dy*dz*velG.x +
                    (1-dx)*dy*dz*velH.x;

            vel.y = (1-dx)*(1-dy)*(1-dz)*velA.y +
                    dx*(1-dy)*(1-dz)*velB.y + 
                    dx*dy*(1-dz)*velC.y +
                    (1-dx)*dy*(1-dz)*velD.y +
                    (1-dx)*(1-dy)*dz*velE.y + 
                    dx*(1-dy)*dz*velF.y + 
                    dx*dy*dz*velG.y +
                    (1-dx)*dy*dz*velH.y;

            vel.z = (1-dx)*(1-dy)*(1-dz)*velA.y +
                    dx*(1-dy)*(1-dz)*velB.y + 
                    dx*dy*(1-dz)*velC.y +
                    (1-dx)*dy*(1-dz)*velD.y +
                    (1-dx)*(1-dy)*dz*velE.y + 
                    dx*(1-dy)*dz*velF.y + 
                    dx*dy*dz*velG.y +
                    (1-dx)*dy*dz*velH.y;
            
            //Store in memory
            partVel[tid] = vel;
        }

    }
    
    __global__ void interpolation3(float4 *partPos, float4 *vertxVel, 
                                  float4 *partVel, int numberFloats)
    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;

        if (tid < numberFloats){
            //Position of the particle
            float4 pos = partPos[tid];
            float dx = pos.x;
            float dy = pos.y;
            float dz = pos.z;


            //Velocities of vertices
            float4 velA = vertxVel[0];
            float4 velB = vertxVel[1];
            float4 velC = vertxVel[2];
            float4 velD = vertxVel[3];
            float4 velE = vertxVel[4];
            float4 velF = vertxVel[5];
            float4 velG = vertxVel[6];
            float4 velH = vertxVel[7];

            float3 velA3 = make_float3(velA.x, velA.y, velA.z);
            float3 velB3 = make_float3(velB.x, velB.y, velB.z);
            float3 velC3 = make_float3(velC.x, velC.y, velC.z);
            float3 velD3 = make_float3(velD.x, velD.y, velD.z);
            float3 velE3 = make_float3(velE.x, velE.y, velE.z);
            float3 velF3 = make_float3(velF.x, velF.y, velF.z);
            float3 velG3 = make_float3(velG.x, velG.y, velG.z);
            float3 velH3 = make_float3(velH.x, velH.y, velH.z);

            //Interpolate velocity 
            float3 vel;

            vel   = (1-dx)*(1-dy)*(1-dz)*velA3 +
                    dx*(1-dy)*(1-dz)*velB3 + 
                    dx*dy*(1-dz)*velC3 +
                    (1-dx)*dy*(1-dz)*velD3 +
                    (1-dx)*(1-dy)*dz*velE3 + 
                    dx*(1-dy)*dz*velF3 + 
                    dx*dy*dz*velG3 +
                    (1-dx)*dy*dz*velH3;

            //Store in memory
            partVel[tid] = make_float4(vel, 0);
        }

    }

    __global__ void test_integrateSimple(float *oldPosX, float *oldPosY, float *oldPosZ, 
                                          float *oldVelu, float *oldVelv, float *oldVelw, 
                                          float *newPosX, float *newPosY, float *newPosZ, 
                                          float deltaTime,
                                          float ximin, float ximax,
                                          float yimin, float yimax,
                                          float zimin, float zimax,
                                          int numberParticles)

    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;

        if (tid < numberParticles){
            float posX1 = oldPosX[tid];
            float posY1 = oldPosY[tid];
            float posZ1 = oldPosZ[tid];

            float velU1 = oldVelu[tid];
            float velV1 = oldVelv[tid];
            float velW1 = oldVelw[tid];

                        //Boundary Conditions
            if(posX1 < ximin)
                posX1 = -posX1;
            if(posX1 > ximax)
                posX1 = ximax - (posX1 - ximax);

            if(posY1 < yimin)
                posY1 = -posY1;
            if(posY1 > yimax)
                posY1 = yimax - (posY1 - yimax);

            if(posZ1 < zimin)
                posZ1 = -posZ1;
            if(posZ1 > zimax)
                posZ1 = zimax - (posZ1 - zimax);
            
            //Update positions
            posX1+= velU1*deltaTime;
            posY1+= velV1*deltaTime;
            posZ1+= velW1*deltaTime;

            //Store in memory
            newPosX[tid] = posX1;
            newPosY[tid] = posY1;
            newPosZ[tid] = posZ1;
        }
    }
        __global__ void integratePeriodic(float *oldPosX, float *oldPosY, float *oldPosZ, 
                                           float *oldVelu, float *oldVelv, float *oldVelw, 
                                           float *newPosX, float *newPosY, float *newPosZ, 
                                           float deltaTime,
                                           float ximin, float ximax,
                                           float yimin, float yimax,
                                           float zimin, float zimax,
                                           int numberParticles)

    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;

        if (tid < numberParticles){
            float posX1 = oldPosX[tid];
            float posY1 = oldPosY[tid];
            float posZ1 = oldPosZ[tid];

            float velU1 = oldVelu[tid];
            float velV1 = oldVelv[tid];
            float velW1 = oldVelw[tid];

            //Boundary Conditions
            if(posX1 < ximin)
                posX1 = ximax + posX1;            
            if(posX1 > ximax)
                posX1 = ximin+ (ximax-posX1);
            
            if(posY1 < yimin)
                posY1 = yimax + posY1;            
            if(posY1 > yimax)
                posY1 = yimin+ (yimax-posY1);


            if(posZ1 < zimin)
                posZ1 = zimax + posZ1;            
            if(posZ1 > zimax)
                posZ1 = zimin+ (zimax-posZ1);
            
            //Update positions
            posX1+= velU1*deltaTime;
            posY1+= velV1*deltaTime;
            posZ1+= velW1*deltaTime;

            //Store in memory
            newPosX[tid] = posX1;
            newPosY[tid] = posY1;
            newPosZ[tid] = posZ1;
        }
    }

    __global__ void integrateBomex(float *oldPosX, float *oldPosY, float *oldPosZ, 
                                   float *oldVelu, float *oldVelv, float *oldVelw, 
                                   float *newPosX, float *newPosY, float *newPosZ, 
                                   float deltaTime,
                                   float ximin, float ximax,
                                   float yimin, float yimax,
                                   float zimin, float zimax,
                                   int numberParticles)

    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;

        if (tid < numberParticles){
            float posX1 = oldPosX[tid];
            float posY1 = oldPosY[tid];
            float posZ1 = oldPosZ[tid];

            float velU1 = oldVelu[tid];
            float velV1 = oldVelv[tid];
            float velW1 = oldVelw[tid];

                        //Boundary Conditions
            if(posX1 < ximin)
                posX1 = ximax + posX1;            
            if(posX1 > ximax)
                posX1 = ximin+ (ximax-posX1);
            
            if(posY1 < yimin)
                posY1 = yimax + posY1;            
            if(posY1 > yimax)
                posY1 = yimin+ (yimax-posY1);


            if(posZ1 < zimin)
                posZ1 = -posZ1;
            if(posZ1 > zimax)
                posZ1 = zimax - (posZ1 - zimax);
            
            //Update positions
            posX1+= velU1*deltaTime;
            posY1+= velV1*deltaTime;
            posZ1+= velW1*deltaTime;

            //Store in memory
            newPosX[tid] = posX1;
            newPosY[tid] = posY1;
            newPosZ[tid] = posZ1;
        }
    }



    __global__ void integrateSimple1(float *oldPosX, float *oldPosY, float *oldPosZ, 
                                     float *oldVelu, float *oldVelv, float *oldVelw, 
                                     float *newPosX, float *newPosY, float *newPosZ, 
                                     float deltaTime,
                                     float ximin, float ximax,
                                     float yimin, float yimax,
                                     float zimin, float zimax,
                                     int numberParticles)

    {
        //Each thread reads, computes, and stores 4 coordinates
        //tid= 0, 4, 8 .... 307196
        int tid = 4*blockIdx.x * gridDim.x + 4*threadIdx.x;

        if (tid < numberParticles){
            float posX1 = oldPosX[tid];
            float posX2 = oldPosX[tid+1];
            float posX3 = oldPosX[tid+2];
            float posX4 = oldPosX[tid+3];
            float posY1 = oldPosY[tid];
            float posY2 = oldPosY[tid+1];
            float posY3 = oldPosY[tid+2];
            float posY4 = oldPosY[tid+3];
            float posZ1 = oldPosZ[tid];
            float posZ2 = oldPosZ[tid+1];
            float posZ3 = oldPosZ[tid+2];
            float posZ4 = oldPosZ[tid+3];


            float velU1 = oldVelu[tid];
            float velU2 = oldVelu[tid+1];
            float velU3 = oldVelu[tid+2];
            float velU4 = oldVelu[tid+3];
            float velV1 = oldVelv[tid];
            float velV2 = oldVelv[tid+1];
            float velV3 = oldVelv[tid+2];
            float velV4 = oldVelv[tid+3];
            float velW1 = oldVelw[tid];
            float velW2 = oldVelw[tid+1];
            float velW3 = oldVelw[tid+2];
            float velW4 = oldVelw[tid+3];
            
            //Update positions
            posX1+= velU1*deltaTime;
            posX2+= velU2*deltaTime;
            posX3+= velU3*deltaTime;
            posX4+= velU4*deltaTime;
            posY1+= velV1*deltaTime;
            posY2+= velV2*deltaTime;
            posY3+= velV3*deltaTime;
            posY4+= velV4*deltaTime;
            posZ1+= velW1*deltaTime;
            posZ2+= velW2*deltaTime;
            posZ3+= velW3*deltaTime;
            posZ4+= velW4*deltaTime;

            //Boundary Conditions
            if(posX1 < ximin)
                posX1 = -posX1;
            if(posX2 < ximin)
                posX2 = -posX2;
            if(posX3 < ximin)
                posX3 = -posX3;
            if(posX4 < ximin)
                posX4 = -posX4;
            if(posX1 > ximax)
                posX1 = ximax - (posX1 - ximax);
            if(posX2 > ximax)
                posX2 = ximax - (posX2 - ximax);
            if(posX3 > ximax)
                posX3 = ximax - (posX3 - ximax);
            if(posX4 > ximax)
                posX4 = ximax - (posX4 - ximax);

            if(posY1 < yimin)
                posY1 = -posY1;
            if(posY2 < yimin)
                posY2 = -posY2;
            if(posY3 < yimin)
                posY3 = -posY3;
            if(posY4 < yimin)
                posY4 = -posY4;
            if(posY1 > yimax)
                posY1 = yimax - (posY1 - yimax);
            if(posY2 > yimax)
                posY2 = yimax - (posY2 - yimax);
            if(posY3 > yimax)
                posY3 = yimax - (posY3 - yimax);
            if(posY4 > yimax)
                posY4 = yimax - (posY4 - yimax);

            if(posZ1 < zimin)
                posZ1 = -posZ1;
            if(posZ2 < zimin)
                posZ2 = -posZ2;
            if(posZ3 < zimin)
                posZ3 = -posZ3;
            if(posZ4 < zimin)
                posZ4 = -posZ4;
            if(posZ1 > zimax)
                posZ1 = zimax - (posZ1 - zimax);
            if(posZ2 > zimax)
                posZ2 = zimax - (posZ2 - zimax);
            if(posZ3 > zimax)
                posZ3 = zimax - (posZ3 - zimax);
            if(posZ4 > zimax)
                posZ4 = zimax - (posZ4 - zimax);

            //Store in memory
            newPosX[tid] = posX1;
            newPosX[tid+1] = posX2;
            newPosX[tid+2] = posX3;
            newPosX[tid+3] = posX4;
            newPosY[tid] = posY1;
            newPosY[tid+1] = posY2;
            newPosY[tid+2] = posY3;
            newPosY[tid+3] = posY4;
            newPosZ[tid] = posZ1;
            newPosZ[tid+1] = posZ2;
            newPosZ[tid+2] = posZ3;
            newPosZ[tid+3] = posZ4;
        }
    }

    //These are some prototypes that aren't ready to be used

    __global__ void test_integrator3(float4 *oldPos, float4 *newPos, 
                                    float4 *oldVel, float4 *newVel, 
                                    float deltaTime, 
                                    int numberFloats)

    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;        
        if (tid < numberFloats/4){
            float4 pos4 = oldPos[tid];
            float4 vel4 = oldVel[tid];

            float3 pos3 = make_float3(pos4.x,pos4.y,pos4.z);
            float3 vel3 = make_float3(pos4.x,pos4.y,pos4.z);

            //Update positions
            pos3+= vel3*deltaTime;

            //Store in memory 
            newPos[tid] = make_float4(pos3,pos4.w);
            newVel[tid] = make_float4(vel3,vel4.w);
        }
    }

    __global__ void test_integrator4(float4 *oldPos, float4 *newPos, 
                                    float4 *oldVel, float4 *newVel, 
                                    float deltaTime, 
                                    int numberFloats)

    {
        int tid = blockIdx.x * gridDim.x + threadIdx.x;        
        if (tid < numberFloats/4){
            float4 pos = oldPos[tid];
            float4 vel = oldVel[tid];
            //Update positions
            //pos+= vel*deltaTime;
            pos.x += vel.x*deltaTime;
            pos.y += vel.y*deltaTime;
            pos.z += vel.z*deltaTime;
            pos.w += vel.w*deltaTime;

            newPos[tid] = pos;
            newVel[tid] = vel;
        }
    }

    //Leo:
    //The function needs to be __global__ otherwise function not found
    //for example: this will fail 
    //void test_rand(int numRand, float *samples)
    __global__ void test_rand(int numRand, float *samples)
    {
        int threadID = blockIdx.x * gridDim.x + threadIdx.x;
        thrust::default_random_engine rng;
        rng.discard(numRand * threadID);
        thrust::uniform_real_distribution<float> rand01(0,1);
        float acc=0.0;
        for (int i=0; i < numRand; i++) {
            float r = rand01(rng);
            acc += r;
        }
        samples[threadID] = acc/numRand; // Normalize back to range [0,1)
    }

    __global__ void initRandVector(float *vector, float minVal, float maxVal, int numFloats)
    {
        int tid = blockIdx.x*gridDim.x + threadIdx.x;
        thrust::default_random_engine rng;
        rng.discard(numFloats*tid);
        thrust::uniform_real_distribution<float> randPos(minVal, maxVal);
        if (tid < numFloats){
            vector[tid]=randPos(rng);
        }
    }

    /*
    * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
    *
    * Please refer to the NVIDIA end user license agreement (EULA) associated
    * with this source code for terms and conditions that govern your use of
    * this software. Any use, reproduction, disclosure, or distribution of
    * this software and related documentation outside the terms of the EULA
    * is strictly prohibited.
    *
    */

   /*
   This file contains simple wrapper functions that call the CUDA kernels
   */

    //Leo:
    //This example was taken from NVIDIA Examples
    //this is a host function and can't be declared with __global__
    //__global__ void
    void
    integrateSystem(float4 *oldPos, float4 *newPos,
                    float4 *oldVel, float4 *newVel,
                    float deltaTime,
                    int numParticles)
    {   
        //Leo: this is done in the device
        thrust::device_ptr<float4> d_newPos(newPos);
        thrust::device_ptr<float4> d_newVel(newVel);
        thrust::device_ptr<float4> d_oldPos(oldPos);
        thrust::device_ptr<float4> d_oldVel(oldVel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_newPos, d_newVel, d_oldPos, d_oldVel)),
            thrust::make_zip_iterator(thrust::make_tuple(d_newPos+numParticles, d_newVel+numParticles, d_oldPos+numParticles, d_oldVel+numParticles)),
            integrate_functor(deltaTime));
    }

    void sortParticles(float *sortKeys, uint *indices, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<float>(sortKeys),
                            thrust::device_ptr<float>(sortKeys + numParticles),
                            thrust::device_ptr<uint>(indices));
    }

}   // extern "C"
