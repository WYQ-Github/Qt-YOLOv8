#include "yolov8_obb_onnx.h"
//using namespace std;
//using namespace cv;
//using namespace cv::dnn;
using namespace Ort;




bool Yolov8ObbOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	if (_batchSize < 1) _batchSize = 1;
	try
	{
		if (!CheckModelPath(modelPath))
			return false;
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
		

		if (isCuda && (cuda_available == available_providers.end()))
		{
			log("Your ORT build without GPU. Change to CPU.");
			log("Infer model on CPU!");
		}
		else if (isCuda && (cuda_available != available_providers.end()))
		{
			log("Infer model on GPU!");
#if ORT_API_VERSION < ORT_OLD_VISON
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = cudaID;
			_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
#else
			OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
#endif
		}
		else
		{
			log("Infer model on CPU!");
		}
		//

		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
		_OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		//init input
		_inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_inputName = _OrtSession->GetInputName(0, allocator);
		_inputNodeNames.push_back(_inputName);
#else
		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
		_inputNodeNames.push_back(_inputName.get());
#endif
		//std::cout << _inputNodeNames[0] << std::endl;
		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		_inputNodeDataType = input_tensor_info.GetElementType();
		_inputTensorShape = input_tensor_info.GetShape();

		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_output_name0 = _OrtSession->GetOutputName(0, allocator);
		_outputNodeNames.push_back(_output_name0);
#else
		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
		_outputNodeNames.push_back(_output_name0.get());
#endif
		Ort::TypeInfo type_info_output0(nullptr);
		type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();
		//warm up
		if (isCuda && warmUp) {
			//draw run
			log("Start warming up");
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}

			delete[]temp;
		}
	}
	catch (const std::exception&) {
		return false;
	}
	return true;

}

int Yolov8ObbOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
	outSrcImgs.clear();
	cv::Size input_size = cv::Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		cv::Mat temp_img = srcImgs[i];
		cv::Vec4d temp_param = {1,1,0,0};
		if (temp_img.size() != input_size) {
			cv::Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		}
		else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	int lack_num =  _batchSize- srcImgs.size();
	if (lack_num > 0) {
		for (int i = 0; i < lack_num; ++i) {
			cv::Mat temp_img = cv::Mat::zeros(input_size, CV_8UC3);
			cv::Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;

}
bool Yolov8ObbOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) {
	std::vector<cv::Mat> input_data = { srcImg };
	std::vector<std::vector<OutputParams>> tenp_output;
	if (OnnxBatchDetect(input_data, tenp_output)) {
		output = tenp_output[0];
		return true;
	}
	else return false;
}
bool Yolov8ObbOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) {
	std::vector<cv::Vec4d> params;
	std::vector<cv::Mat> input_images;
	cv::Size input_size(_netWidth, _netHeight);
	//preprocessing
	Preprocessing(srcImgs, input_images, params);
	cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, cv::Scalar(0, 0, 0), true, false);

	int64_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

	output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		input_tensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);
	//post-process
	float* all_data = output_tensors[0].GetTensorMutableData<float>();
	_outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	int net_width = _outputTensorShape[1];
	int socre_array_length = net_width - 5;
	int angle_index = net_width - 1;
	int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
	for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
		cv::Mat output0 = cv::Mat(cv::Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
		all_data += one_output_length;
		float* pdata = (float*)output0.data;
		int rows = output0.rows;
		std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::RotatedRect> boxes;
		for (int r = 0; r < rows; ++r) {    
			cv::Mat scores(1, socre_array_length, CV_32F, pdata + 4);
			cv::Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= _classThreshold) {

				//rect [x,y,w,h]
				float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
				float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
				float w = pdata[2] / params[img_index][0];  //w
				float h = pdata[3] / params[img_index][1];  //h
				float angle = pdata[angle_index] / CV_PI * 180.0;
				class_ids.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre);
				boxes.push_back(cv::RotatedRect(cv::Point2f(x, y), cv::Size(w, h), angle));

			}
			pdata += net_width;
		}

		std::vector<int> nms_result;
		cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
		std::vector<std::vector<float>> temp_mask_proposals;
		std::vector<OutputParams> temp_output;
		for (int i = 0; i < nms_result.size(); ++i) {
			int idx = nms_result[i];
			OutputParams result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];
			result.rotatedBox = boxes[idx];
			if (result.rotatedBox.size.width < 1 || result.rotatedBox.size.height < 1)
				continue;
			temp_output.push_back(result);
		}
		output.push_back(temp_output);
	}

	if (output.size())
		return true;
	else
		return false;
}