__author__ = 'pressel'


import time
import numpy as np
import scipy.stats as stats
import matplotlib.cm as cm
import pylab as plt
import sys

from Namelist import Namelist
from Parallel import Parallel
from Grid import Grid
from PrognosticVariables import PrognosticVariables
from BasicState import BasicState
from InitializeFactory import initialize_factory
from SurfaceFactory import surface_factory
from ThermodynamicsFactory import thermodynamics_factory
from MicrophysicsFactory import microphysics_factory
from ForcingFactory import forcing_factory
from ScalarAdvectionFactory import scalaradvection_factory
from MomentumAdvectionFactory import momentumadvection_factory
from SGSFactory import sgs_factory
from ScalarDiffusion2nd import ScalarDiffusion2nd
from MomentumDiffusion2nd import MomentumDiffusion2nd
from PressureFactory import PressureFactory
from TimeSteppingFactory import timestepping_factory
from StatisticalOutput import StatisticalOutput
from DampingFactory import damping_factory
from RadiationFactory import radiation_factory
from Output import Output
#Lagrangian particles
#from LagrangianParticles import LagrangianParticles
#from LagrangianParticlesGPU1 import LagrangianParticles as LagrangianParticlesGPU
from LagrangianParticlesGPU import LagrangianParticles as LagrangianParticlesGPU

from TimeSteppingManager import TimeSteppingManager

from Diagnostics import Diagnostics

from Parallel import reduce_to_root

class Simulation:


    def __init__(self):
        self.nml =  Namelist()

        #Open the case file
        case_file_handle = open(self.nml.case_file,'rb')
        self.case_dict = np.ma.load(case_file_handle)
        case_file_handle.close()

        self.comm = Parallel(self.nml)
        self.grid = Grid(self.comm, self.nml)
        ''' IOhdf5Statistics must be imported after Parallel is initialized in order to avoid conflicts related to
        multiple calls to mpi_init'''
        from IOhdf5 import IOhdf5Statistics
        self.io = IOhdf5Statistics(self.nml,self.comm,self.grid)
        self.statisticaloutput = StatisticalOutput()
        self.velocities = PrognosticVariables()
        self.scalars = PrognosticVariables()
        self.pressure = PrognosticVariables()
        self.timestepping = timestepping_factory(self.case_dict,self.nml)
        self.timestepping_manager = TimeSteppingManager()
        self.basicstate = BasicState(self.grid)
        self.init= initialize_factory(self.case_dict,self.nml)
        self.thermodynamics = thermodynamics_factory(self.nml,self.grid,self.comm)
        self.microphysics = microphysics_factory(self.grid,self.nml)
        self.radiation = radiation_factory(self.grid,self.nml)
        self.forcing = forcing_factory(self.case_dict,self.nml,self.comm)
        self.surface = surface_factory(self.case_dict,self.nml,self.comm)
        self.scalaradvection = scalaradvection_factory(self.nml,self.grid)
        self.momentumadvection = momentumadvection_factory(self.nml,self.grid)
        self.sgs = sgs_factory(self.nml)
        self.scalardiffusion = ScalarDiffusion2nd(self.grid)
        self.momentumdiffusion = MomentumDiffusion2nd(self.grid)
        self.poisson = PressureFactory(self.nml,self.grid)
        self.damping = damping_factory(self.case_dict,self.nml,self.grid)

        #For Lagrangian Particles
        #self.particles = LagrangianParticles()
        self.particlesGPU = LagrangianParticlesGPU()
        
        self.diagnostics = Diagnostics(self.nml,self.case_dict)
        self.output = Output()

    def initialize(self):

        #Add specific momentum (aka velocity) components as prognostic variables
        self.velocities.add_variable('u','symmetric',units='m s^-1')
        self.velocities.add_variable('v','symmetric',units='m s^-1')
        self.velocities.add_variable('w','antisymmetric',units='m s^-1')

        #Add entropy and total water mixing ration as prognostic variables
        self.scalars.add_variable('specific_entropy','symmetric',units='')
        self.scalars.add_variable('y_water','symmetric',units='kg kg^-1')

        self.microphysics.initialize(self.grid,self.scalars,self.io)

        #Initialize the pressure field
        self.pressure.add_variable('p','symmetric',units='Pa')
        self.sgs.add_prognostics(self.scalars)

        #Now initialize the prognostic variable arrays
        self.velocities.initialize(self.grid)
        self.scalars.initialize(self.grid)
        self.pressure.initialize(self.grid)

        self.statisticaloutput.initialize(self.grid,self.scalars,self.velocities,self.io)

        self.timestepping.initialize(self.scalars,self.velocities,self.grid)
        self.timestepping_manager.initialize(self.nml)

        self.init.initialize(self.case_dict, self.grid, self.basicstate, self.velocities, self.scalars,self.io,self.comm)


        self.scalars.update_boundary_conditions(self.grid,self.comm)
        self.velocities.update_boundary_conditions(self.grid,self.comm)

        self.scalars.compute_mean_profiles(self.grid,self.comm)
        self.velocities.compute_mean_profiles(self.grid,self.comm)

        self.thermodynamics.initialize(self.grid,self.io)

        self.radiation.initialize(self.case_dict,self.grid,self.scalars, self.thermodynamics,self.io)



        self.forcing.initialize(self.case_dict,self.basicstate,self.grid,self.scalars,self.velocities,self.thermodynamics,self.io)

        self.scalaradvection.initialize(self.nml,self.grid, self.scalars)
        self.momentumadvection.initialize(self.grid, self.velocities)

        self.sgs.initialize(self.case_dict,self.scalars,self.grid,self.io)
        self.scalardiffusion.initialize(self.grid, self.scalars,self.io)
        self.momentumdiffusion.initialize(self.grid,self.velocities,self.io)


        self.surface.initialize(self.case_dict,self.comm,self.grid,self.basicstate,self.scalars,self.velocities,self.io)


        self.poisson.initialize(self.grid)

        self.damping.initialize(self.case_dict,self.grid)

        #Lagragian particles 
        #self.particles.initialize(self.grid,self.case_dict,self.nml,self.comm)
        self.particlesGPU.initialize(self.grid,self.case_dict,self.nml,self.comm)

        self.diagnostics.initialize(self.scalars,self.velocities,self.grid,self.io)

        self.output.initialize()

        #Free memory from things that we will no longer use
        self.case_dict = None
        self.nml = None
        self.init = None


    def run(self):
        self.thermodynamics.update(self.grid,self.basicstate,self.scalars, self.velocities,self.comm,self.io)
        self.velocities.zero_all_tendencies()
        self.scalars.zero_all_tendencies()
        while(self.timestepping.time < self.timestepping.timemax):
            tic = time.time()
            for self.timestepping.rk_step in range(self.timestepping.num_rk_step):
                self.output.update(self.basicstate,self.grid,self.velocities,self.scalars,
                        self.thermodynamics,self.microphysics,self.forcing,self.radiation,self.surface
                        ,self.scalaradvection,self.momentumadvection,self.sgs,self.scalardiffusion,self.momentumdiffusion,
                        self.statisticaloutput,self.diagnostics,self.timestepping,self.io,self.comm)


                self.velocities.compute_mean_profiles(self.grid,self.comm)
                self.scalars.compute_mean_profiles(self.grid,self.comm)
                self.thermodynamics.update(self.grid,self.basicstate,self.scalars, self.velocities,self.comm,self.io)
                self.microphysics.update(self.grid, self.basicstate, self.thermodynamics, self.scalars,
                                         self.velocities,self.timestepping,self.comm,self.io)
                self.forcing.update(self.grid,self.basicstate,self.scalars,self.velocities,self.thermodynamics,
                                    self.timestepping,self.comm,self.io)
                self.radiation.update(self.grid,self.basicstate,self.scalars,self.velocities,self.thermodynamics,
                                      self.surface,self.comm,self.io,self.timestepping)
                self.surface.update(self.grid,self.basicstate,self.scalars,self.velocities,self.thermodynamics,self.radiation,self.timestepping,
                                    self.comm,self.io)
                self.scalaradvection.update( self.grid, self.basicstate, self.scalars, self.velocities)
                self.momentumadvection.update( self.grid, self.basicstate, self.scalars, self.velocities)
                self.sgs.update(self.grid, self.basicstate, self.scalars, self.velocities, self.thermodynamics,
                                self.timestepping, self.surface, self.comm, self.io)
                self.scalardiffusion.update( self.grid, self.basicstate, self.scalars, self.velocities,
                                           self.sgs,self.surface,self.comm,self.io)
                self.momentumdiffusion.update(self.grid, self.basicstate, self.scalars, self.velocities,
                                              self.sgs,self.surface,self.comm,self.io)
                self.damping.update(self.grid,self.scalars,self.velocities)


                self.timestepping.update(self.scalars,self.velocities,self.grid,self.io,self.comm)


                self.poisson.update(self.comm,self.grid, self.basicstate, self.scalars, self.velocities, self.pressure,
                                       self.timestepping)
                self.scalars.update_boundary_conditions(self.grid,self.comm)
                self.velocities.update_boundary_conditions(self.grid,self.comm)

                self.timestepping_manager.update(self.grid,self.velocities,self.timestepping,self.comm,self.io)
                
                #Lagrangian particles                
                #self.particles.update(self.grid, self.scalars, self.velocities, self.timestepping, self.comm)
                self.particlesGPU.update(self.grid, self.scalars, self.velocities, self.timestepping, self.comm)


            #reduced_data = reduce_to_root(self.thermodynamics.potential_temperature,self.grid,self.comm)
            toc = time.time()
            
            if(self.comm.rank == 0):
                print('Time: ',self.timestepping.time, 'dt: ', self.timestepping.dt, ' walltime: ', toc - tic, 'CFL-Max:', self.timestepping.cfl_max)
                #s_fluc = self.velocities.values[:,:,:,2] #- s_mean[np.newaxis,np.newaxis,:]
                #plt.figure(0,figsize=(12,4))
                #levels = np.linspace(284.9,300.1,250)
                #plt.contourf(reduced_data[:,5,:].T,levels)#s_fluc[self.grid.gw:self.grid.nxl-self.grid.gw,5,self.grid.gw:self.grid.nzl-self.grid.gw].T,levels=levels,cmap=cm.RdBu_r)
                #plt.gca().set_aspect('equal', 'box')
                #levels = np.linspace(0.0,0.001,10)
                #if(np.max(np.max(self.thermodynamics.y_liquid[self.grid.gw:-self.grid.gw,10,self.grid.gw:-self.grid.gw]))>0.0):
                #    plt.contour(self.thermodynamics.y_liquid[self.grid.gw:-self.grid.gw,10,self.grid.gw:-self.grid.gw].T,levels=levels)
                #plt.clim(-3.0,3.0)
                #plt.clim(284.9,300.1)
                #plt.savefig('./figs/'+str(1000000+np.int(self.timestepping.time)) + '.png')
                #plt.close()
                #del reduced_data
            

    def finalize(self):
        pass
