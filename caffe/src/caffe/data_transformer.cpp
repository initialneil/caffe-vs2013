#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
    Blob<Dtype>* transformed_blob, const Dtype* mean) {
    const int mat_num = mat_vector.size();
    const int num = transformed_blob->num();
    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    
    CHECK_GT(mat_num, 0) << "There is no MAT to add";
    CHECK_EQ(mat_num, num) <<
        "The size of mat_vector must be equals to transformed_blob->num()";
    Blob<Dtype> uni_blob(1, channels, height, width);
    for (int item_id = 0; item_id < mat_num; ++item_id) {
        int offset = transformed_blob->offset(item_id);
        uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
        Transform(mat_vector[item_id], &uni_blob, mean);    
    }
}

#ifndef OSX
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
    Blob<Dtype>* transformed_blob, const Dtype* mean) {
    const int img_channels = cv_img.channels();
    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;

    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    const int num = transformed_blob->num();

    CHECK_EQ(channels, img_channels);
    CHECK_LE(height, img_height);
    CHECK_LE(width, img_width);
    CHECK_GE(num, 1);

    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

    const int crop_size = param_.crop_size();
    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && Rand(2);
    const bool has_mean_file = param_.has_mean_file();
    // mean_values_ is not supported right now
    vector<Dtype> mean_values_;
    const bool has_mean_values = mean_values_.size() > 0;

    CHECK_GT(img_channels, 0);
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);

    //Dtype* mean = NULL;
    if (has_mean_file) {
        //CHECK_EQ(img_channels, data_mean_.channels());
        //CHECK_EQ(img_height, data_mean_.height());
        //CHECK_EQ(img_width, data_mean_.width());
        //mean = data_mean_.mutable_cpu_data();
    }
    if (has_mean_values) {
        CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
            "Specify either 1 mean_value or as many as channels: " << img_channels;
        if (img_channels > 1 && mean_values_.size() == 1) {
            // Replicate the mean_value for simplicity
            for (int c = 1; c < img_channels; ++c) {
                mean_values_.push_back(mean_values_[0]);
            }
        }
    }

    int h_off = 0;
    int w_off = 0;
    cv::Mat cv_cropped_img = cv_img;
    if (crop_size) {
        CHECK_EQ(crop_size, height);
        CHECK_EQ(crop_size, width);
        // We only do random crop when we do training.
        if (phase_ == Caffe::TRAIN) {
            h_off = Rand(img_height - crop_size + 1);
            w_off = Rand(img_width - crop_size + 1);
        } else {
            h_off = (img_height - crop_size) / 2;
            w_off = (img_width - crop_size) / 2;
        }
        cv::Rect roi(w_off, h_off, crop_size, crop_size);
        cv_cropped_img = cv_img(roi);
    } else {
        CHECK_EQ(img_height, height);
        CHECK_EQ(img_width, width);
    }

    CHECK(cv_cropped_img.data);

    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    int top_index;
    for (int h = 0; h < height; ++h) {
        const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < img_channels; ++c) {
                if (do_mirror) {
                    top_index = (c * height + h) * width + (width - 1 - w);
                } else {
                    top_index = (c * height + h) * width + w;
                }
                // int top_index = (c * height + h) * width + w;
                Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                if (has_mean_file) {
                    //int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
                    int mean_index = (c * height + h + h_off) * width + w + w_off;
                    transformed_data[top_index] =
                        (pixel - mean[mean_index]) * scale;
                } else {
                    if (has_mean_values) {
                        transformed_data[top_index] =
                            (pixel - mean_values_[c]) * scale;
                    } else {
                        transformed_data[top_index] = pixel * scale;
                    }
                }
            }
        }
    }
}
#endif

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
    CHECK(rng_);
    CHECK_GT(n, 0);
    caffe::rng_t* rng =
        static_cast<caffe::rng_t*>(rng_->generator());
    return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
