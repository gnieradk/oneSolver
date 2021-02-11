```bash
# To compile a C++ sycl program for NVIDIa
# tutorial is for ubuntu 20.04, should work for 18.04 too

# Install CUDA 10.2
# https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# Install ONEapi
# https://software.intel.com/content/www/us/en/develop/articles/installing-intel-oneapi-toolkits-via-apt.html
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
sudo apt update
sudo apt install intel-basekit

# Install llvm sycl from source
# https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
export CUDA_HOME=/usr/local/cuda-10.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export DPCPP_HOME=~/sycl_workspace
mkdir $DPCPP_HOME
cd $DPCPP_HOME

git clone https://github.com/intel/llvm -b sycl
python $DPCPP_HOME/llvm/buildbot/configure.py --cuda
python $DPCPP_HOME/llvm/buildbot/compile.py


# to build your program
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$DPCPP_HOME/llvm/build/lib

. /opt/intel/oneapi/setvars.sh
export PATH=$DPCPP_HOME/sycl_workspace/llvm/build/bin/:$PATH

# build your program using clang++
# do not forget to add flags: -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice

# Run your program
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
src/one-solver-exhaustive --input ../benchmarks/exhaustive_search/examples/mvp24.qubo --output /tmp/test.csv --device-type gpu

```
