cd ../../../
set PROTOC=../3rdparty/tools/protoc.exe

echo caffe.pb.h is being generated
"%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe.proto"

copy /y "src\\caffe\\proto\\caffe.pb.h" "proto\\caffe.pb.h"

echo caffe_pretty_print.pb.h is being generated
"%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe_pretty_print.proto"

copy /y "src\\caffe\\proto\\caffe_pretty_print.pb.h" "proto\\caffe_pretty_print.pb.h"

pause

