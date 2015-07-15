cd ../../../
set PROTOC=../3rdparty/tools/protoc.exe

if exist "src/caffe/proto/caffe.pb.h" (
    echo caffe.pb.h remains the same as before
) else (
    echo caffe.pb.h is being generated
    "%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe.proto"
)

if exist "src/caffe/proto/caffe_pretty_print.pb.h" (
    echo caffe_pretty_print.pb.h remains the same as before
) else (
    echo caffe_pretty_print.pb.h is being generated
    "%PROTOC%" -I="src/caffe/proto" --cpp_out="src/caffe/proto/" "src/caffe/proto/caffe_pretty_print.proto"
)


