
#this is veeery important !!!!! .pydistutils.cfg
module load open_mpi 
module load hdf5

local_path3=$HOME/local3
mkdir $local_path3

wget --no-check-certificate  https://www.python.org/ftp/python/3.4.1/Python-3.4.1.tgz
tar -zxvf Python-3.4.1.tgz
cd Python-3.4.1 
./configure --prefix=$local_path3
make install 
cd ../
alias python3=$local_path3/bin/python3.4

#Install cython 
python3 -m pip install cython 

#install numpy 
python3 -m pip install numpy 

#install scipy 
python3 -m pip install scipy 

#install mpi4py 
python3 -m pip install mpi4py  

#install h5py 
python3 -m pip install h5py 

#install matplot
python3 -m pip install matplotlib

#install PyCUDA
python3 -m pip install pycuda

#on euler install manually six, dateutil, pyparsing
