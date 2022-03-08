from distutils.core import setup
import distutils.core
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import mpi4py as mpi4py
import sys, subprocess, os



#Get platform specific compile and linking arguments. 
#Will use distutils.core.sys.platform to determine the system
#architecture
extra_compile_args=[]
extra_link_args=[]
platform = distutils.core.sys.platform

args = sys.argv[1:]

if "cleanall" in sys.argv:
    print("Deleting cython files...")
    subprocess.Popen("rm -rf *.c", shell = True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so", shell =True, executable="/bin/bash")
    subprocess.Popen("rm -rf fig/*.png", shell = True, executable ="bin/bash")
    sys.argv.remove("cleanall")
    print(sys.argv)

elif "-gpu" in sys.argv:
    import pycuda.autoinit, pycuda.driver
    from pycuda.compiler import compile
    import numpy
    import time
    import os 

    #Loading cudaCode.cu
    print("Compiling CUDA modules with PyCUDA")
    codeFile = open("particleSystem_cuda.cu","r+")
    code = codeFile.read()
    
    #Compiling and saving CUBIN file
    binaryStr=compile(code,no_extern_c=True, include_dirs=[os.getcwd(),os.getcwd()+'/inc'])
    binFile = open("particleSystem_cuda.cubin","wb")
    binFile.write(binaryStr)
    binFile.close()

    sys.argv.remove("-gpu")

#addinf --inplace to arguments
if "build_ext" in args:
    if args.count('--inplace')==0:
        sys.argv.append('--inplace')
        print("adding --inplace")

    if platform == 'darwin':
        #optimizer for new gcc (4.8)
        #    extra_compile_args+=['-O3','-march=native','-Wno-unused','-Wno-#warnings']
        extra_compile_args+=['-O3','-Wno-unused','-Wno-#warnings']
    
    elif platform =='linux2':
        extra_compile_args+=['-O3','-openmp']
        extra_link_args+=['-openmp']


    #Now get include paths from relevant python modules
    include_path = [np.get_include()]
    include_path += [mpi4py.get_include()]
    
    #Now actually configure the extension build process
    extensions = [
        Extension("*", ["*.pyx"],
                    include_dirs = include_path,
                    libraries = [],
                    library_dirs = [],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args)
    ]
    setup(
        name = "My hello app",
        ext_modules = cythonize(extensions),
    )

