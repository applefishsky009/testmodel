#include <string>
#include <vector>
#include <iostream>
#include "caffe/caffe.hpp"
#include "caffe/caffe_reg.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
using namespace caffe;

// 前向传播
void caffe_forward(	boost::shared_ptr<Net<float>> &net, 
					float *data_ptr) {
	Blob<float>* input_blobs = net->input_blobs()[0];
	switch (Caffe::mode()) {
	case Caffe::CPU:
		memcpy(input_blobs, data_ptr, 
			sizeof(float) * input_blobs->count());
		break;
	case Caffe::GPU:
		cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,
			sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
		break;
	default:
		LOG(FATAL) << "Unknown Caffe mode";
	}
	net->ForwardPrefilled();
}

// 根据blob_name获取其在网络中的index
unsigned int get_blob_index(boost::shared_ptr<Net<float>> &net, char *query_blob_name) {
	std::string str_query(query_blob_name);
	vector<string> const &blob_names = net->blob_names();
	for (int i = 0; i < blob_names.size(); ++i) {
		if (str_query == blob_names[i]) {
			return i;
		}
	}
	LOG(FATAL) << "Unknown blob name: " << str_query;
}

// 根据layer名字获取其在网络中的Index
unsigned int get_layer_index(boost::shared_ptr<Net<float>> &net, char *query_layer_name) {
	std::string str_query(query_layer_name);
	vector<string> const &layer_names = net->layer_names();
	for (unsigned int i = 0; i < layer_names.size(); ++i) {
		if (str_query == layer_names[i]) {
			return i;
		}
	}
	LOG(FATAL) << "Unknown layer name:" << str_query;
}

int main() {
	// 初始化网络
	char *proto = "D:/visual/testmodel/testmodel/2021-01-14-07-16-39_crf_net.prototxt";
	Phase phase = TEST;	/* or TRAIN */
	//Caffe::set_mode(Caffe::CPU);
	Caffe::set_mode(Caffe::GPU);
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(proto, phase));

	// 加载已经训练好的模型
	char *model = "D:/visual/testmodel/testmodel/2021-01-14-07-16-39_crf_netn.caffemodel";
	net->CopyTrainedLayersFrom(model);

	// 读取模型中的每层结构配置参数
	NetParameter param;
	ReadNetParamsFromBinaryFileOrDie(model, &param);
	int num_layers = param.layer_size();
	for (int i = 0; i < num_layers; ++i) {
		// 结构体配置参数：name kernel_size pad stride
		// 和yangshicai的demo不同, 参数在protobuf中被序列化了
		LOG(ERROR) << "Layer " << i << ":" << param.layer(i).name() << "\t" << param.layer(i).type();
		if (param.layer(i).type() == "Convolution") {
			ConvolutionParameter conv_param = param.layer(i).convolution_param();
			LOG(ERROR) << "\t\tkernel_size: " << conv_param.kernel_size().size()
				<< ", pad: " << conv_param.pad().size()
				<< ", stride: " << conv_param.stride().size();
		}
	}
	cout << "********************" << endl;
	int kkk = 0;
	// 读取网络指定feature层数据 blobs
	{
		char *query_blob_name = "conv1";
		unsigned int blob_id = get_blob_index(net, query_blob_name);
		boost::shared_ptr<Blob<float>>  blob = net->blobs()[blob_id];
		unsigned int num_data = blob->count();	// 
		const float *blob_ptr = (const float *)blob->cpu_data();
	}

	// 指定layer的权重数据
	// !Note:不同于Net的Blob是Feature Maps, Layer的Blob是指Conv和FC等层的weight和Bias
	{
		char *query_layer_name = "conv1";
		const float *weight_ptr, *bias_ptr;
		unsigned int layer_id = get_layer_index(net, query_layer_name);
		boost::shared_ptr<Layer<float>> layer = net->layers()[layer_id];
		std::vector<boost::shared_ptr<Blob<float>>> blobs = layer->blobs();
		if (blobs.size() > 0) {
			weight_ptr = (const float *)blobs[0]->cpu_data();
			bias_ptr = (const float *)blobs[1]->cpu_data();
		}
	}

	// 修改某层的weight数据
	{
		const float* data_ptr = NULL;		// 指向待写入数据的指针, 原数据指针
		float *weight_ptr = NULL;	// 指向网络中某层权重的值, 目标数据指针
		unsigned int data_size = 0;		// 待写入的数据量
		char *layer_name = "conv1";	// 需要修改的layer名字
		unsigned int layer_id = get_layer_index(net, layer_name);
		boost::shared_ptr<Blob<float>> blob = net->layers()[layer_id]->blobs()[0];
		CHECK(data_size == blob->count());	// 待写入的数据量和参数数据量一致
		switch (Caffe::mode()) {
		case Caffe::CPU:
			weight_ptr = blob->mutable_cpu_data();
			break;
		case Caffe::GPU:
			weight_ptr = blob->mutable_gpu_data();
			break;
		default:
			LOG(FATAL) << "Unknown Caffe mode";
		}
		caffe_copy(blob->count(), data_ptr, weight_ptr);
	}

	// 保存新的模型
	char *weights_file = "bvlc_reference_caffenet_new.caffemodel";
	NetParameter net_param;	// 初始化权值参数
	net->ToProto(&net_param, false);	// 存储网络权值
	WriteProtoToBinaryFile(net_param, weights_file);	// 保存权值
	return 0;
}

