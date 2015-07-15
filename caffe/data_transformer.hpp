#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "opencv2/opencv.hpp"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param)
    : param_(param) {
    phase_ = Caffe::phase();
  }
  virtual ~DataTransformer() {}

  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param batch_item_id
   *    Datum position within the batch. This is used to compute the
   *    writing position in the top blob's data
   * @param datum
   *    Datum containing the data to be transformed.
   * @param mean
   * @param transformed_data
   *    This is meant to be the top blob's data. The transformed data will be
   *    written at the appropriate place within the blob's data.
   */
  void Transform(const int batch_item_id, const Datum& datum,
                 const Dtype* mean, Dtype* transformed_data);

  /**
  * @brief Applies the transformation defined in the data layer's
     * transform_param block to a vector of Mat.
     *
     * @param mat_vector
     *    A vector of Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See memory_layer.cpp for an example.
     */
#ifndef OSX
    void Transform(const vector<cv::Mat> & mat_vector,
        Blob<Dtype>* transformed_blob, const Dtype* mean);
#endif

    /**
    * @brief Applies the transformation defined in the data layer's
    * transform_param block to a cv::Mat
    *
    * @param cv_img
    *    cv::Mat containing the data to be transformed.
    * @param transformed_blob
    *    This is destination blob. It can be part of top blob's data if
    *    set_cpu_data() is used. See image_data_layer.cpp for an example.
    */
#ifndef OSX
    void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob, const Dtype* mean);
#endif

 protected:
  virtual unsigned int Rand();
  virtual int Rand(int n);

  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Caffe::Phase phase_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

