set DATA=../../data/mnist
set TOOLS=../../../3rdparty/tools

REM set BACKEND=lmdb
set BACKEND=leveldb

echo "Creating %BACKEND%..."

rd /s /q "mnist_train_%BACKEND%"
rd /s /q "mnist_test_%BACKEND%"

"%TOOLS%/convert_mnist_data.exe" %DATA%/train-images-idx3-ubyte %DATA%/train-labels-idx1-ubyte mnist_train_%BACKEND% --backend=%BACKEND%
"%TOOLS%/convert_mnist_data.exe" %DATA%/t10k-images-idx3-ubyte %DATA%/t10k-labels-idx1-ubyte mnist_test_%BACKEND% --backend=%BACKEND%

echo "Done."

pause