__author__ = 'ilciavo'

import numpy as np
#cimport numpy as np
import numpy as np
from mpi4py import MPI
import cython
import pylab as plt
import pycuda.autoinit, pycuda.driver
from pycuda.compiler import SourceModule 
mod = pycuda.driver.module_from_file("particleSystem_cuda.cubin")
import pycuda.gpuarray as gpuarray

#import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

#Arrow3d is a child of FancyArrowPatch
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

class LagrangianParticles:

    def __init__(self):

        self.local_ximin = None
        self.local_ximax = None
        self.local_yimin = None
        self.local_yimax = None
        self.local_zimax = None
        self.local_zimin = None

        self.local_xcmin = None
        self.local_xcmax = None
        self.local_ycmin = None
        self.local_ycmax = None
        self.local_zcmax = None
        self.local_zcmin = None

        self.values = None   #Will be initialized as a numpy array
        self.names = None    #Will contain a list of particle names corresponding to their values

        self.num_particles_local = None
        self.num_particle_values = None

        self.count = 0

        self.oldPosX_dev=None
        self.oldPosY_dev=None
        self.oldPosZ_dev=None
        self.partVelu_dev=None
        self.partVelv_dev=None
        self.partVelw_dev=None
       
        self.blocks=0
        #threads run in warps of 32 threads
        self.threads_per_block=32


        return

    def initialize(self,grid,case_dict,nml,comm):

        gw = grid.gw
        nxl = grid.nxl
        nyl = grid.nyl
        nzl = grid.nzl

        #Getting the max and min coordinates for an interface
        #Leo: index correction
        self.local_ximin = grid.xi[gw-1]
        self.local_ximax = grid.xi[nxl-gw-1]
        self.local_yimin = grid.yi[gw-1]
        self.local_yimax = grid.yi[nyl-gw-1]
        self.local_zimin = grid.zi[gw-1]
        self.local_zimax = grid.zi[nzl-gw-1]

        #Getting the max and min coordinates for a cell
        #Leo:index  correction
        self.local_xcmin = grid.xc[gw]
        self.local_xcmax = grid.xc[nxl-gw-1]
        self.local_ycmin = grid.yc[gw]
        self.local_ycmax = grid.yc[nyl-gw-1]
        self.local_zcmin = grid.zc[gw]
        self.local_zcmax = grid.zc[nzl-gw-1]

        #numParticles = Sim.particles.num_particles_local
        #no more than 1024
        #numParticles = 256*4
        #numParticles = 256/64
        self.num_particles_local = nml.numberParticles
        self.threads_per_block=nml.threads_per_block
        
        #threads_per_block = 32
        #blocks = 512
        if self.num_particles_local%self.threads_per_block == 0:
            self.blocks = int((self.num_particles_local)/self.threads_per_block)
        else:
            self.blocks = int((self.num_particles_local)//self.threads_per_block+1)
    
        threads = self.threads_per_block*self.blocks

        self.oldPosX_dev = gpuarray.zeros(self.num_particles_local, dtype=np.float32)
        self.oldPosY_dev = gpuarray.zeros(self.num_particles_local, dtype=np.float32)
        self.oldPosZ_dev = gpuarray.zeros(self.num_particles_local, dtype=np.float32)


        print('Initializing ', self.num_particles_local,' Particles on GPU using', threads,'threads distributed on ',self.blocks,'blocks and ',self.threads_per_block,'threads/block')

        init = mod.get_function('initRandVectors')

        init(self.oldPosX_dev,
             self.oldPosY_dev,
             self.oldPosZ_dev, 
             np.float32(self.local_ximin),
             np.float32(self.local_ximax),
             np.float32(self.local_yimin),
             np.float32(self.local_yimax), 
             np.float32(self.local_zimin),
             np.float32(self.local_zimax), 
             np.int32(self.num_particles_local,),
             grid=(self.blocks,1), block=(self.threads_per_block,1,1))
        
        self.partVelu_dev = gpuarray.zeros(self.num_particles_local, dtype=np.float32)
        self.partVelv_dev = gpuarray.zeros(self.num_particles_local, dtype=np.float32)
        self.partVelw_dev = gpuarray.zeros(self.num_particles_local, dtype=np.float32)

        return


    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def update(self,grid,scalars,velocities,timestepping,comm):

        #values = self.values
        num_particles_local = self.num_particles_local

        xc = np.float32(grid.xc)
        yc = np.float32(grid.yc)
        zc = np.float32(grid.zc)

        xi = np.float32(grid.xi)
        yi = np.float32(grid.yi)
        zi = np.float32(grid.zi)

        nx = grid.nx
        ny = grid.ny
        nz = grid.nz

        gw = grid.gw
        dx = grid.dx
        dy = grid.dy
        dz = grid.dz

        udof = velocities.get_dof('u')
        vdof = velocities.get_dof('v')
        wdof = velocities.get_dof('w')
        u = velocities.values[:,:,:,udof]
        v = velocities.values[:,:,:,vdof]
        w = velocities.values[:,:,:,wdof]

        u_1d=np.float32(u.reshape(u.size))
        v_1d=np.float32(v.reshape(v.size))
        w_1d=np.float32(w.reshape(w.size))
        
        dt =  timestepping.dt

        bottom_boundary = grid.bottom_boundary
        top_boundary = grid.top_boundary

        #Update particle positions
        sending = False
        
        #GPU Interpolation
        interpolate = mod.get_function('interpolation')

        #Copying values to GPU everyiteration
        #Using values from GPU memory 
        interpolate(self.oldPosX_dev,
                    self.oldPosY_dev,
                    self.oldPosZ_dev,
                    pycuda.driver.In(xi),
                    pycuda.driver.In(yi),
                    pycuda.driver.In(zi),
                    pycuda.driver.In(u_1d),
                    pycuda.driver.In(v_1d),
                    pycuda.driver.In(w_1d),
                    self.partVelu_dev,
                    self.partVelv_dev,
                    self.partVelw_dev,
                    np.float32(self.local_ximin),
                    np.float32(self.local_ximax),
                    np.float32(self.local_yimin),
                    np.float32(self.local_yimax), 
                    np.float32(self.local_zimin),
                    np.float32(self.local_zimax), 
                    np.float32(nx), np.float32(ny), np.float32(nz), 
                    np.int32(gw),
                    np.float32(dx), np.float32(dy), np.float32(dz), 
                    np.int32(num_particles_local),
                    grid=(self.blocks,1), block=(self.threads_per_block,1,1)) 


        #Integrating velocities for particles
        #integrate1 = mod.get_function('integrateSimple')
        #integrate1 = mod.get_function('integratePeriodic')
        integrate1 = mod.get_function('integrateBomex')
        

        #Copying values to GPU every iteration
        newPosX_dev= gpuarray.zeros(self.num_particles_local, dtype=np.float32)
        newPosY_dev= gpuarray.zeros(self.num_particles_local, dtype=np.float32)
        newPosZ_dev= gpuarray.zeros(self.num_particles_local, dtype=np.float32)

        #Using stored values in the GPU 
        integrate1(self.oldPosX_dev,
                   self.oldPosY_dev,
                   self.oldPosZ_dev,
                   self.partVelu_dev,
                   self.partVelv_dev,
                   self.partVelw_dev,
                   newPosX_dev,
                   newPosY_dev,
                   newPosZ_dev,
                   np.float32(dt),
                   np.float32(self.local_ximin),
                   np.float32(self.local_ximax),
                   np.float32(self.local_yimin),
                   np.float32(self.local_yimax),
                   np.float32(self.local_zimin),
                   np.float32(self.local_zimax),
                   np.int32(self.num_particles_local),
                   grid=(self.blocks,1), block=(self.threads_per_block,1,1))
        
        #Plotting Interpolated velocities for particles
        fig = plt.figure(figsize=(16,9))
        #ax = plt.axes(projection='3d')        
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.oldPosX_dev.get(), self.oldPosY_dev.get(), self.oldPosZ_dev.get(),'.', markersize=5, color='red', alpha=0.5)
        ax.plot(newPosX_dev.get(), newPosY_dev.get(), newPosZ_dev.get(),'.', markersize=5, color='red', alpha=1)        
        ax.set_xlabel('x_values')
        ax.set_ylabel('y_values')
        ax.set_zlabel('z_values')
        ax.set_xlim([self.local_ximin,self.local_ximax])
        ax.set_ylim([self.local_yimin,self.local_yimax])
        ax.set_zlim([self.local_zimin,self.local_zimax])
        #plt.show()
        plt.savefig('./figs/'+str(1000000+np.int(timestepping.time)) + '.png')
        plt.close()
        
        self.oldPosX_dev=newPosX_dev.copy()
        self.oldPosY_dev=newPosY_dev.copy()
        self.oldPosZ_dev=newPosZ_dev.copy()

        return

    def output(self):
        pass
