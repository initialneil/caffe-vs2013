echo "Downloading..."

set wget="../../../3rdparty/tools/wget.exe"
set do_7za="../../../3rdparty/tools/7za.exe"

%wget% --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
%wget% --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
%wget% --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
%wget% --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "Unzipping..."

%do_7za% x train-images-idx3-ubyte.gz
%do_7za% x train-labels-idx1-ubyte.gz
%do_7za% x t10k-images-idx3-ubyte.gz
%do_7za% x t10k-labels-idx1-ubyte.gz

REM Creation is split out because leveldb sometimes causes segfault
REM and needs to be re-created.

echo "Renaming..."

rename train-images.idx3-ubyte train-images-idx3-ubyte
rename train-labels.idx1-ubyte train-labels-idx1-ubyte
rename t10k-images.idx3-ubyte t10k-images-idx3-ubyte
rename t10k-labels.idx1-ubyte t10k-labels-idx1-ubyte

echo "Done."
