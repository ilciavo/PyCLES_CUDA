#this is veeery important !!!!! .pydistutils.cfg
module load open_mpi 
module load hdf5

local_path2=$HOME/local2

mkdir $local_path2

wget --no-check-certificate  https://www.python.org/ftp/python/2.7.2/Python-2.7.2.tgz
tar -zxvf Python-2.7.2.tgz
cd Python-2.7.8 
./configure --prefix=$local_path2
make install 
cd ../
alias python2=$local_path2/bin/python2.7

wget --no-check-certificate https://pypi.python.org/packages/source/s/setuptools/setuptools-5.7.tar.gz#md5=81f980854a239d60d074d6ba052e21ed
tar -zxvf setuptools-5.7.tar.gz
cd setuptools-5.7
python2 setup.py build 
#if you get an error "no module named io" fixing the hidden file ~/.pyutil...
python2 setup.py install
cd ../
alias easy_install=$local_path2/bin/easy_install
easy_install pip

#Install cython 
python2 -m pip install cython 

#install numpy 
python2 -m pip install numpy 

#install scipy 
python2 -m pip install scipy 

#install mpi4py 
python2 -m pip install mpi4py  

#install h5py 
python2 -m pip install h5py 

#install matplot
python2 -m pip install matplotlib

#install PyCUDA
python2 -m pip install pycuda

#on euler install manually six, dateutil, pyparsing   
