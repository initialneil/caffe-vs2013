REM go to the caffe root
cd ../../

set BIN=../build/x64/Release

"%BIN%/caffe.exe" train --solver=examples/mnist/lenet_solver-lmdb.prototxt

pause