rm -rf ./build/
mkdir build
cd ./build
cmake ..
make
./evaluate_accuracy ../compiled_by_H_.xmodel /root/jupyter_notebooks/pynq-croos/HPatches/i_ajuntament/