cd ../../../
set PROTOC=../3rdparty/tools/protoc.exe

echo caffe.pb.h is being generated
"%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe.proto"

echo caffe_pretty_print.pb.h is being generated
"%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe_pretty_print.proto"

pause

