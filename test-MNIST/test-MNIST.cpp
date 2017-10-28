/*
* Developer : Neil Z. SHAO (initialneil@gmail.com)
* Date : 16/07/2015
*
* Load trained Caffe model and run single image test with OpenCV
*/

//添加注释，用于github测试  胡2017年10月28日20:29:45

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/smart_ptr/shared_ptr.hpp"

// Caffe's required library
#pragma comment(lib, "caffe.lib")

// enable namespace
using namespace std;
using namespace cv;
using namespace boost;
using namespace caffe;

// set caffe root path manually
const string CAFFE_ROOT = "../caffe";

int main(int argc, char** argv)
{
    // get a testing image and display
    Mat img = imread(CAFFE_ROOT + "/examples/images/mnist_5.png");
    cvtColor(img, img, CV_BGR2GRAY);
    imshow("img", img);
    waitKey(1);

    // Set up Caffe
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    Caffe::SetDevice(device_id);
    LOG(INFO) << "Using GPU";

    // Load net
    Net<float> net(CAFFE_ROOT + "/examples/mnist/lenet_test-memory-1.prototxt");
    string model_file = CAFFE_ROOT + "/examples/mnist/lenet_iter_10000.caffemodel";
    net.CopyTrainedLayersFrom(model_file);

    // set the patch for testing
    vector<Mat> patches;
    patches.push_back(img);

    // push vector<Mat> to data layer
    float loss = 0.0;
    boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
    memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float>>(net.layer_by_name("data"));
    
    vector<int> labels(patches.size());
    memory_data_layer->AddMatVector(patches, labels);

    // Net forward
    const vector<Blob<float>*> & results = net.ForwardPrefilled(&loss);
    float *output = results[1]->mutable_cpu_data();

    // Display the output
    for (int i = 0; i < 10; i++) {
        printf("Probability to be Number %d is %.3f\n", i, output[i]);
    }
    waitKey(0);
}
