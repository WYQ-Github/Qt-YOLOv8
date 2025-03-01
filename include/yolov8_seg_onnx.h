#pragma once
#include <iostream>
#include<memory>
#include <opencv2/opencv.hpp>
#include "yolov8_utils.h"
#include<onnxruntime_cxx_api.h>
//#include <tensorrt_provider_factory.h>  //if use OrtTensorRTProviderOptionsV2
//#include <onnxruntime_c_api.h>

class Yolov8SegOnnx {
public:
	Yolov8SegOnnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~Yolov8SegOnnx() {
		if (_OrtSession != nullptr)
			delete _OrtSession;
	};// delete _OrtMemoryInfo;


public:
	/** \brief Read onnx-model
	* \param[in] modelPath:onnx-model path
	* \param[in] isCuda:if true,use Ort-GPU,else run it on cpu.
	* \param[in] cudaID:if isCuda==true,run Ort-GPU on cudaID.
	* \param[in] warmUp:if isCuda==true,warm up GPU-model.
	*/
	bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true);

	/** \brief  detect.
	* \param[in] srcImg:a 3-channels image.
	* \param[out] output:detection results of input image.
	*/
	bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output);
	/** \brief  detect,batch size= _batchSize
	* \param[in] srcImg:A batch of images.
	* \param[out] output:detection results of input images.
	*/
	bool OnnxBatchDetect(std::vector<cv::Mat>& srcImg, std::vector<std::vector<OutputParams>>& output);

private:

	template <typename T>
	T VectorProduct(const std::vector<T>& v)
	{
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};
	int PreProcessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params);

	const int _netWidth = 640;   //ONNX-net-input-width
	const int _netHeight = 640;  //ONNX-net-input-height

	int _batchSize = 1;  //if multi-batch,set this
	bool _isDynamicShape = false;//onnx support dynamic shape
	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;


	//ONNXRUNTIME	
	Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov8");
	Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
	Ort::Session* _OrtSession = nullptr;
	Ort::MemoryInfo _OrtMemoryInfo;
#if ORT_API_VERSION < ORT_OLD_VISON

	char* _inputName, * _output_name0, * _output_name1;
#else
	std::shared_ptr<char> _inputName, _output_name0,_output_name1;
#endif

	std::vector<char*> _inputNodeNames;
	std::vector<char*> _outputNodeNames;

	size_t _inputNodesNum = 0;      
	size_t _outputNodesNum = 0;     

	ONNXTensorElementDataType _inputNodeDataType; 
	ONNXTensorElementDataType _outputNodeDataType;
	std::vector<int64_t> _inputTensorShape; 

	std::vector<int64_t> _outputTensorShape;
	std::vector<int64_t> _outputMaskTensorShape;

	std::function<void(const std::string&)> m_logCallback;

	void log(const std::string& message) {
		if (m_logCallback) {
			m_logCallback(message);
		}
	}
public:
	std::vector<std::string> _className = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	void setLogCallback(std::function<void(const std::string&)> callback) {
        m_logCallback = callback;
    }

};