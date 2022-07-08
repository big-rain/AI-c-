// YolactPre.cpp: 定义应用程序的入口点。
//

#include "YolactPre.h"

using namespace std;


std::string DEVICE = "CPU";


void yolactPredictor::sigmoid(Mat& out, int length)
{
	float* pdata = (float*)(out.data);
	int i = 0;
	for (i = 0; i < length; i++) {
		pdata[i] = 1.0 / (1 + exp(-pdata[i]));
	}
}

void yolactPredictor::make_priors() {
	this->num_priors = 0;
	int p = 0;
	for (p = 0; p < 5; p++) {
		this->num_priors += this->conv_ws[p] * this->conv_hs[p] * 3;
	}

	std::cout << "num_priors=" << this -> num_priors << std::endl;
	this->priorbox = new float[4 * this -> num_priors];
	// generate priorbox 
	float* pb = priorbox;
	for (p = 0; p < 5; p++) {
		int conv_w = this->conv_ws[p];
		int conv_h = this->conv_hs[p];
		
		float scale = this->scales[p];

		for (int i = 0; i < conv_h; i++) {
			for (int j = 0; j < conv_w; j++) {
				// +0.5, because priors are in center=-size nolation
				float cx = (j + 0.5f) / conv_w;
				float cy = (i + 0.5f) / conv_h;

				for (int k = 0; k < 3; k++) {
					float ar = aspect_ratios[k];
					ar = sqrt(ar);

					float w = scale * ar / this->target_size;
					float h = scale / ar / this->target_size;

					// This is for backward compatability with a bug where I made everything square by accident
					// cfg.backbone.use_square_anchors:
					h = w;
					pb[0] = cx;
					pb[1] = cy;
					pb[2] = w;
					pb[3] = h;
					pb += 4;
				}
			}
		}
	}
}


// rle 压缩存储技术: 用一对pair值来表示值为一的mask位置， 
// 比如（3， 3）表示从第三个元素起，后三个元素都为1
// 元素排序，由上往下， 由左到右

std::vector<int> masktorle(cv::Mat& mask) {
	// convert mask to rle
	// 1-mask, 0-background
	int img_w = mask.cols;
	int img_h = mask.rows;
	//printf("%d", img_h);

	vector<float> fmask;
	vector<int> counts;
	
	for (int row = 0; row < img_h; row++) {
		for (int col = 0; col < img_w; col++) {
			float t = mask.at<float>(row, col);
			// printf("%f ", t);
			
		if (t >= 0.5) {
				t = 1;
		  }
		else {
			t = 0;
			}
			fmask.push_back(t);
		}
	}

	int len = 1;
	int size = img_h * img_w;
	for (int count = 0; count < size-1; count++) {
		int temp = fmask[count];
		while ( temp == fmask[++count] && count++ < size-1) {
			len++;
		}
		counts.push_back(len);
		// printf("%d,", len);
		len = 1;
	}
	return counts;
	
}

bool yolactPredictor::loadModel(std::string model_dir_path) {
	_model_dir_path = model_dir_path;

	make_priors();

	// --------  1.  初始化Core --------
	ov::Core core;

	std::cout << "Loading model to the device " << DEVICE << std::endl;
	// -------- 2. 读取模型 --------
	std::cout << "Loading model files:"  << _model_dir_path << std::endl;


	model = core.compile_model(model_dir_path, DEVICE);

//	OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
//	OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");


	// --------------------3.配置网络输入输出

	return true;
}
/*
void fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image) {
	// 获取输入节点要求的输入图片数据的大小
	ov::Shape tensor_shape = input_tensor.get_shape();
	const size_t width = tensor_shape[3]; // 要求输入图片数据的宽度
	const size_t height = tensor_shape[2]; // 要求输入图片数据的高度
	const size_t channels = tensor_shape[1]; // 要求输入图片数据的维度
	// 读取节点数据内存指针
	float* input_tensor_data = input_tensor.data<float>();
	// 将图片数据填充到网络中
	// 原有图片数据为 H、W、C 格式，输入要求的为 C、H、W 格式
	for (size_t c = 0; c < channels; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				input_tensor_data[c * width * height + h * width + w] = input_image.at<cv::Vec<float, 3>>(h, w)[c];
			}
		}
	}
}
*/
// string yolactPredictor::predict(std::string img_path) {
std::vector<yolactPredictor::Object>yolactPredictor::predict(cv::Mat& srcOriginal) {
	int img_w = srcOriginal.cols;
	int img_h = srcOriginal.rows;

	// ov::Core core;

	// string onnx_path = "D:\\CPP_Code\\YolactPre\\model\\yolact_base_54_800000.onnx";
	// ov::CompiledModel model = core.compile_model(onnx_path, DEVICE);
	const ov::Layout model_layout{ "NCHW" };

	/*
	cv::Mat images;
	images = cv::imread(img_path);
	if (images.data == nullptr) {
		std::cout << "unable to read input image file '" << img_path << "'" << std::endl;
	}
	*/


	// ---------创建Infer Request------------------
	std::cout<< "Create infer request" << std::endl;
	ov::InferRequest infer_request = model.create_infer_request();


	ov::Tensor input_tensor = infer_request.get_input_tensor();
	ov::Shape input_shape = input_tensor.get_shape();
	size_t num_channels = input_shape[1];
	size_t image_width = input_shape[2];
	size_t image_height = input_shape[3];
	size_t image_size = image_width * image_height;
// 	printf("%d", image_size);
	// Mat frame = imread(image_file, IMREAD_COLOR);
	// ---------读取图片并按照模型输入要求进行预处理
	// int image_width = srcOriginal.cols;
	// int image_height = srcOriginal.rows;
	// int _max = max(h, w);
	// cv::cvtColor(srcOriginal, srcOriginal, cv::COLOR_BGR2RGB);// 因为yolact模型接收的顺序是RGB的 而opencv的顺序是BGR的 需要进行一下转换再进行yolact模型的图像预处理

	cv::Mat blob_image;//转换为网络可以解析图片格式
	cv::resize(srcOriginal, blob_image, cv::Size(image_height,image_width));//转换大小


	blob_image.convertTo(blob_image, CV_32F);//转换为浮点数
	blob_image = blob_image / 255.0;//转换到0-1之间
	cv::subtract(blob_image, cv::Scalar(0.485, 0.456, 0.406), blob_image);//每个通道的值都减去均值
	cv::divide(blob_image, cv::Scalar(0.229, 0.224, 0.225), blob_image);// 每个通道的值都除以方差



	//NCHW
	float* input_data = input_tensor.data<float>();//转成指针格式，由指针指向每个像素
	//对像素进行遍历
	for (size_t row = 0; row < image_height; row++)
	{
		for (size_t col = 0; col < image_width; col++)
		{
			for (size_t c = 0; c < num_channels; c++)
			{
				input_data[image_size * c + row * image_width + col] = blob_image.at<Vec3f>(row, col)[c];
			}
		}
	}

	infer_request.set_input_tensor(input_tensor);



	// ----------copy NHWC data from image to tensor with batch
/*	unsigned char* image_data_ptr = image_data.get();
	unsigned char* tensor_data_ptr = input_tensor.data<unsigned char>();

	size_t image_size = image_width * image_height * image_channels;

	for (size_t i = 0; i < image_size; i++) {
		tensor_data_ptr[i] = image_data_ptr[i];
	}
	*/
	//----执行推理计算
	auto start = std::chrono::system_clock::now();
	infer_request.infer();
	auto end = std::chrono::system_clock::now();
	std::cout << "inference time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;



	//[1,138,138,32]
	ov::Tensor proto = infer_request.get_tensor("proto");

	//[1,19248,32]
	auto mask = infer_request.get_tensor("mask");
	const float* maskdim = mask.data<const float>();
	cv::Mat maskdim_mat(19248, 32, CV_32FC1, (float*)maskdim);

	// [1,19248,81]
	ov::Tensor confidence = infer_request.get_tensor("conf");
	const float* conf = confidence.data<const float>();
	cv::Mat cof_mat(19248, 81, CV_32FC1, (float*)conf);



	//[1,19248,4]
	ov::Tensor loc = infer_request.get_tensor("loc");
	const float* location = loc.data<const float>();
	cv::Mat loc_mat(19248, 4, CV_32FC1, (void*)location);



	//-------------------------------------------
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> maskIds;
	const int num_class = cof_mat.cols;
	//for (int i = 0; i < this->num_priors; i++) {
	
// 	}
	
	// Prints formatted segmentation results（解析box和mask)

	for (int i = 0; i < this->num_priors; i++) {
		Mat scores = cof_mat.row(i).colRange(1, num_class);
		Point classIdPoint;
		double score;
		// get the value and location of the maxium score
		minMaxLoc(scores, 0, &score, 0, &classIdPoint);
		if (score > 0.5) {
			const float* loc = (float*)loc_mat.data + i * 4;
			const float* pd = this->priorbox + i * 4;
			
			
			float pd_cx = pd[0];
			float pd_cy = pd[1];
			float pd_w = pd[2];
			float pd_h = pd[3];

			float bbox_cx = var[0] * loc[0] * pd_w + pd_cx;
			float bbox_cy = var[1] * loc[1] * pd_h + pd_cy;
			float bbox_w = (float)(exp(var[2] * loc[2]) * pd_w);
			float bbox_h = (float)(exp(var[3] * loc[3]) * pd_h);

			float obj_x1 = bbox_cx - bbox_w * 0.5f;
			float obj_y1 = bbox_cy - bbox_h * 0.5f;
			float obj_x2 = bbox_cx + bbox_w * 0.5f;
			float obj_y2 = bbox_cy + bbox_h * 0.5f;

			// clip
			obj_x1 = max(min(obj_x1 * img_w,  (float)(img_w  - 1)), 0.f);
			obj_y1 = max(min(obj_y1 * img_h, (float)(img_h - 1)), 0.f);
			obj_x2 = max(min(obj_x2 * img_w,  (float)(img_w  - 1)), 0.f);
			obj_y2 = max(min(obj_y2 * img_h, (float)(img_h - 1)), 0.f);

			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
			boxes.push_back(Rect((int)obj_x1, (int)obj_y1, (int)(obj_x2 - obj_x1 + 1), (int)(obj_y2 - obj_y1 + 1)));
			maskIds.push_back(i);

		//	std::cout << "class" << classIdPoint.x << std::endl;
		//	std::cout << "maskIds:" << i << std::endl;
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidence
	vector<int> indices;
	vector<yolactPredictor::Object> instances;
	yolactPredictor::Object instance;

	dnn::NMSBoxes(boxes, confidences, 0.5, 0.5, indices, 1.f, this->_keep_top_k);

	for (size_t i = 0; i < indices.size(); ++i) {
		int idx = indices[i];
		// printf("%d\n", idx);
		Rect box = boxes[idx];
		
		int xmax = box.x + box.width;
		int ymax = box.y + box.height;
		cv::Rect coordinate(box.x, box.y, xmax, ymax);

		instance.rect = coordinate;
	

		printf("\n\n坐标：x:%d,  y:%d, w:%d, h:%d", box.x, box.y, xmax, ymax);
		rectangle(srcOriginal, Point(box.x, box.y), Point(xmax, ymax), Scalar(0, 0, 255), 3);
		//printf("draw mask\n");
		// get the label for the class name and its confidence
		char text[256];
		//printf("%s:", class_names[classIds[idx] + 1]);
		sprintf(text, "%s: %.2f", class_names[classIds[idx] + 1], confidences[idx]);

		instance.name = class_names[classIds[idx] + 1];
		instance.scores = confidences[idx];
		puts(text);

		// Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 1, &baseLine);

		int ymin = max(box.y, labelSize.height);
		putText(srcOriginal, text, Point(box.x, ymin), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);



		Mat mask(this->mask_h, this->mask_w, CV_32FC1);
		mask = cv::Scalar(0.f);

		int channel = maskdim_mat.cols;
		int area = this->mask_h * this->mask_w;
		
		//printf("%d",maskIds[idx]);
		float* coeff = (float*)maskdim_mat.data + maskIds[idx] * channel;
		
		float* pm = (float*)mask.data;

		// [1, 138, 138, 32]
		ov::Tensor proto = infer_request.get_tensor("proto");
		const float* pmaskmap = proto.data<const float>();

		for (int j = 0; j < area; j++) {
			for (int p = 0; p < channel; p++) {
				pm[j] += pmaskmap[p] * coeff[p];
			}
			pmaskmap += channel;
		}

		this->sigmoid(mask, area);
		Mat mask2;
		resize(mask, mask2, Size(img_w, img_h));

		vector<int> rle = masktorle(mask2);
		instance.rle_mask = rle;
		// int num_c = mask2.channels();
		// printf("通道数 %d", num_c);
		instances.push_back(instance);

		for (int y = 0; y < img_h; y++) {
			const float* pmask = (float*)mask2.data + y * img_w;
			// printf("%daaaa\n", (float*)mask2.data[4]);
			uchar* p = srcOriginal.data + y * img_w * 3;
			for (int x = 0; x < img_w; x++) {
				if (pmask[x] > 0.5) {
					// printf("%f ", pmask[x]);
					p[0] = (uchar)(p[0] * 0.5 + colors[classIds[idx] + 1][0] * 0.5);
					p[1] = (uchar)(p[1] * 0.5 + colors[classIds[idx] + 1][1] * 0.5);
					p[2] = (uchar)(p[2] * 0.5 + colors[classIds[idx] + 1][2] * 0.5);
				}
				p += 3;
			}
		}

	}
	
	return instances;
}




void yolactPredictor::setupPredictor(std::string config) {
	return;
}
string yolactPredictor::predict(std::string img_path) {
	return("0");
}

/*
int main() {
	
	yolactPredictor yolact;
	bool result = yolact.loadModel("D:\\CPP_Code\\YolactPre\\model\\yolact_base_54_800000.onnx");
	cv::Mat img = cv::imread("D:\\CPP_Code\\YolactPre\\data\\04.jpg", IMREAD_COLOR);
	std::vector<yolactPredictor::Object> predict;
	
	predict = yolact.predict(img);
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, img);
	waitKey(0);
	destroyAllWindows();
	//ov::Core core;
	return 0;
	
	//std::cout << "hello world!" << std::endl;
}
*/
