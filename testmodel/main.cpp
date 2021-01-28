#include <string>
#include <vector>
#include <iostream>
#include "caffe/caffe.hpp"
#include "caffe/caffe_reg.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
using namespace caffe;

// ǰ�򴫲�
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

// ����blob_name��ȡ���������е�index
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

// ����layer���ֻ�ȡ���������е�Index
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
	// ��ʼ������
	char *proto = "D:/visual/testmodel/testmodel/2021-01-14-07-16-39_crf_net.prototxt";
	Phase phase = TEST;	/* or TRAIN */
	//Caffe::set_mode(Caffe::CPU);
	Caffe::set_mode(Caffe::GPU);
	boost::shared_ptr<Net<float>> net(new caffe::Net<float>(proto, phase));

	// �����Ѿ�ѵ���õ�ģ��
	char *model = "D:/visual/testmodel/testmodel/2021-01-14-07-16-39_crf_netn.caffemodel";
	net->CopyTrainedLayersFrom(model);

	// ��ȡģ���е�ÿ��ṹ���ò���
	NetParameter param;
	ReadNetParamsFromBinaryFileOrDie(model, &param);
	int num_layers = param.layer_size();
	for (int i = 0; i < num_layers; ++i) {
		// �ṹ�����ò�����name kernel_size pad stride
		// ��yangshicai��demo��ͬ, ������protobuf�б����л���
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
	// ��ȡ����ָ��feature������ blobs
	{
		char *query_blob_name = "conv1";
		unsigned int blob_id = get_blob_index(net, query_blob_name);
		boost::shared_ptr<Blob<float>>  blob = net->blobs()[blob_id];
		unsigned int num_data = blob->count();	// 
		const float *blob_ptr = (const float *)blob->cpu_data();
	}

	// ָ��layer��Ȩ������
	// !Note:��ͬ��Net��Blob��Feature Maps, Layer��Blob��ָConv��FC�Ȳ��weight��Bias
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

	// �޸�ĳ���weight����
	{
		const float* data_ptr = NULL;		// ָ���д�����ݵ�ָ��, ԭ����ָ��
		float *weight_ptr = NULL;	// ָ��������ĳ��Ȩ�ص�ֵ, Ŀ������ָ��
		unsigned int data_size = 0;		// ��д���������
		char *layer_name = "conv1";	// ��Ҫ�޸ĵ�layer����
		unsigned int layer_id = get_layer_index(net, layer_name);
		boost::shared_ptr<Blob<float>> blob = net->layers()[layer_id]->blobs()[0];
		CHECK(data_size == blob->count());	// ��д����������Ͳ���������һ��
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

	// �����µ�ģ��
	char *weights_file = "bvlc_reference_caffenet_new.caffemodel";
	NetParameter net_param;	// ��ʼ��Ȩֵ����
	net->ToProto(&net_param, false);	// �洢����Ȩֵ
	WriteProtoToBinaryFile(net_param, weights_file);	// ����Ȩֵ
	return 0;
}

