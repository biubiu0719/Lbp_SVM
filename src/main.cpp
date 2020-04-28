#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>  
#include <time.h>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <stdio.h>
#include<sys/time.h>
#define BAD_TRAIN_FILES_0704 6322
#define BAD_TRAIN_FILES_0711 5736
#define BAD_TRAIN_FILES_0715 28000
#define GOOD_TRAIN_FILES_0704 3161
#define GOOD_TRAIN_FILES_0711 2868
#define GOOD_TRAIN_FILES_0715 13000
#define GOOD_TRAIN_FILES_0303 19000
#define BAD_TRAIN_FILES_0303 39000
#define HARD_TRAIN_FILES 0
#define CSIZE 16
#define PSIZE 48
using namespace std;
using namespace cv;
using namespace ml;
#define UNTRAIN
void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage);
void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);

	Mat LBPImage;
	ComputeLBPImage_Uniform(srcImage, LBPImage);

	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;

	// 特征向量的个数
	int numberOfDimension = 58 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));

	// 计算LBP特征向量
	int stepOfCell = srcImage.cols;
	int index = -58;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector = (float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index += 58;

			// 计算每个cell的LBP直方图
			Mat cell = LBPImage(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell = cell.data;
			int sum = 0; // 每个cell的等价模式总数
			for (int y_Cell = 0; y_Cell <= cell.rows - 1; ++y_Cell, rowOfCell += stepOfCell)
			{
				uchar *colOfCell = rowOfCell;
				for (int x_Cell = 0; x_Cell <= cell.cols - 1; ++x_Cell, ++colOfCell)
				{
					if (colOfCell[0] != 0)
					{
						// 在直方图中转化为0~57，所以是colOfCell[0] - 1
						++dataOfFeatureVector[index + colOfCell[0] - 1];
						++sum;
					}
				}
			}

			// 一定要归一化！否则分类器计算误差很大
			for (int i = 0; i <= 57; ++i)
				dataOfFeatureVector[index + i] /= sum;

		}
	}
}

// 计算等价模式LBP特征图，为了方便表示特征图，58种等价模式表示为1~58,第59种混合模式表示为0
// 注：你可以将第59类混合模式映射为任意数值，因为要突出等价模式特征，所以非等价模式设置为0比较好
void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
	LBPImage.create(srcImage.size(), srcImage.type());

	// 计算LBP图
	// 扩充原图像边界，便于边界处理
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);

	// 构建LBP 等价模式查找表
	//int table[256];
	//BuildUniformPatternTable(table);

	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

	// 计算LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP = LBPImage.cols;
	uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
	uchar *rowOfLBPImage = LBPImage.data;
	for (int y = 1; y <= heightOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// 计算LBP值
			int LBPValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])
				LBPValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0])
				LBPValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])
				LBPValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0])
				LBPValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])
				LBPValue += 1;

			colOfLBPImage[0] = table[LBPValue];

		} // x

	}// y
}


void get_goodtrain(Mat& trainingFeatures, Mat& trainingLabels)
{
  //  string dir = "../data/train/gooddata/";
  string dir = "../../../data/20190704/positive/";
    vector<string> files;
for(int i=0;i<GOOD_TRAIN_FILES_0704;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190711/positive/";
for(int i=0;i<GOOD_TRAIN_FILES_0711;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190715/positive/";
for(int i=0;i<GOOD_TRAIN_FILES_0715;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20200303/positive/";
for(int i=0;i<GOOD_TRAIN_FILES_0303;i++)
{
        files.push_back(dir + to_string(i) + ".jpg");
}


    
	for (int i = 0; i < files.size(); i++)
	{
		Mat SrcImage = imread(files[i].c_str());
		
		Mat train_data;
		

		resize(SrcImage, train_data, Size(PSIZE, PSIZE));
		Mat imgGray;
		cvtColor(train_data, imgGray, COLOR_BGR2GRAY);
		Mat featureVector;
		ComputeLBPFeatureVector_Uniform(imgGray,Size(CSIZE,CSIZE),featureVector);

		int feature_dim = featureVector.cols;
		if (i == 0) {
			trainingFeatures = Mat::zeros(BAD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0711+BAD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+HARD_TRAIN_FILES+GOOD_TRAIN_FILES_0303+BAD_TRAIN_FILES_0303, feature_dim, CV_32FC1);
			trainingLabels = Mat::zeros(BAD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0711+BAD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+HARD_TRAIN_FILES+GOOD_TRAIN_FILES_0303+BAD_TRAIN_FILES_0303, 1, CV_32SC1);
		}
		float * featurePtr = trainingFeatures.ptr<float>(i);
		int * labelPtr = trainingLabels.ptr<int>(i);
		for (int j = 0; j < feature_dim; j++) {
			*featurePtr = featureVector.at<float>(0,j);
			featurePtr++;
		}
		*labelPtr = 1;
		labelPtr++;
    }
}
void get_badtrain(Mat& trainingFeatures, Mat& trainingLabels)
{
    string dir = "../../../data/20190704/negative_true/";
    vector<string> files;
	for(int i=0;i<BAD_TRAIN_FILES_0704;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190711/negative_true/";
for(int i=0;i<BAD_TRAIN_FILES_0711;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190715/negative_true/";
for(int i=0;i<BAD_TRAIN_FILES_0715;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20200303/negative/";
for(int i=0;i<BAD_TRAIN_FILES_0303;i++)
{
        files.push_back(dir + to_string(i) + ".jpg");
}


	for (int i = 0; i < files.size(); i++)
	{
		Mat SrcImage = imread(files[i].c_str());
		//Mat train_data(64, 128, CV_32FC1);
		Mat train_data;
		//resize(SrcImage, train_data, Size(48, 128));
		resize(SrcImage, train_data, Size(PSIZE, PSIZE));
		Mat imgGray;
		cvtColor(train_data, imgGray, COLOR_BGR2GRAY);
		Mat featureVector;
		ComputeLBPFeatureVector_Uniform(imgGray,Size(CSIZE,CSIZE),featureVector);

		int feature_dim = featureVector.cols;
		float * featurePtr = trainingFeatures.ptr<float>(i+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0303);
		int * labelPtr = trainingLabels.ptr<int>(i+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0303);
		for (int j = 0; j < feature_dim; j++)
         {
			*featurePtr = featureVector.at<float>(0,j);
			featurePtr++;
		}
		*labelPtr = -1;
		labelPtr++;
    }
}
void get_hardtrain(Mat& trainingFeatures, Mat& trainingLabels)
{
    string dir = "../../../data/train/harddata/";
    vector<string> files;
    // for (int i = 0; i < BAD_TRAIN_FILES; i++)								// 取前400张数字0来训练
	// {
	// 	files.push_back(dir + to_string(i) + ".jpg");
	// }
	for(int i=1;i<10;i++)
	{
		files.push_back(dir +"0000"+ to_string(i) + ".jpg");
	}
	for(int i=10;i<=HARD_TRAIN_FILES;i++)
	{
		files.push_back(dir +"000"+ to_string(i) + ".jpg");
	}
	
    //HOGDescriptor *hog = new HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
  HOGDescriptor *hog = new HOGDescriptor(Size(48, 48), Size(8, 8), Size(4, 4), Size(4, 4), 9);
	for (int i = 0; i < files.size(); i++)
	{
		Mat SrcImage = imread(files[i].c_str());
		//Mat train_data(64, 128, CV_32FC1);
		Mat train_data;
		//resize(SrcImage, train_data, Size(64, 128));
		resize(SrcImage, train_data, Size(48, 48));
		vector<float>descriptor;								// hog特征描述子
		//hog->compute(train_data, descriptor, Size(8, 8));		// 计算hog特征
		hog->compute(train_data, descriptor, Size(4, 4));
		int feature_dim = descriptor.size();
		float * featurePtr = trainingFeatures.ptr<float>(i+GOOD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0704);
		int * labelPtr = trainingLabels.ptr<int>(i+GOOD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0704);
		for (int j = 0; j < feature_dim; j++)
         {
			*featurePtr = descriptor[j];
			featurePtr++;
		}
		*labelPtr = -1;
		labelPtr++;
    }
}
int main()
{
	struct timeval start,end;
    //获取训练数据
	string model_path = "lbp_svm_48x48_16x16_LINEAR.xml";

#ifdef TRAIN
    Mat trainingFeatures, trainingLabels;
	cout<<"加载训练图"<<endl;
	get_goodtrain(trainingFeatures, trainingLabels);
	cout<<"正样本加载完成"<<endl;
    get_badtrain(trainingFeatures, trainingLabels);
	cout<<"负样本加载完成"<<endl;

    Ptr<TrainData> data = TrainData::create(trainingFeatures, ROW_SAMPLE, trainingLabels);

    //设置SVM训练器参数并训练
	Ptr<SVM> model = SVM::create();
	model->setDegree(0);
	model->setGamma(1);
	model->setCoef0(1.0);
	model->setC(10);
	model->setNu(0.5);
	model ->setP(1.0);
	model->setType(SVM::C_SVC);														// svm类型
	//model->setKernel(SVM::LINEAR);														// kernel类型，核函数，线性核
	model->setKernel(SVM::LINEAR);	
	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,1000, FLT_EPSILON));
model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER , 1500, FLT_EPSILON));
//model->setTermCriteria(TermCriteria(TermCriteria::EPS, 200, FLT_EPSILON));
    cout<<"开始训练!"<<endl;
    
   model->trainAuto(data);	
    //保存模型

	model->save(model_path);

	cout << "训练完毕！开始测试：" << endl;


//     model->load(model_path);
// model->trainAuto(data);
// model->save(model_save);
#endif
#ifdef UNTRAIN
    Ptr<SVM>model=SVM::load(model_path);//直接获取数据进行检测
#endif

	cout<<"开始检测"<<endl;
    int count0 = 0;//成功检测到人的个数
	string dir = "./../../../data/20200303/positive/";
	
        
	for (int i =19000; i < 24700; i++)								// 取后100张数字0来测试
	{
		Mat testData, testFeatures;
		Mat test = imread(dir + to_string(i) + ".jpg");
		//Mat test_data(64, 128, CV_32FC1);
		Mat test_data;
		
		resize(test, test_data, Size(PSIZE, PSIZE));
		Mat imgGray;
		cvtColor(test_data, imgGray, COLOR_BGR2GRAY);
		Mat featureVector;
		ComputeLBPFeatureVector_Uniform(imgGray,Size(CSIZE,CSIZE),featureVector);

		int feature_dim = featureVector.cols;
		

		testFeatures = Mat::zeros(1, feature_dim, CV_32FC1);
		float * featurePtr = testFeatures.ptr<float>(0);
		for (int j = 0; j < feature_dim; j++) {
			*featurePtr = featureVector.at<float>(0,j);
			featurePtr++;
		}
		int k = model->predict(testFeatures);
		if (k == 1)
			count0++;
	}

    int count1 = 0;//成功检测到不是人的个数
    dir = "./../../../data/20200303/negative/";
	gettimeofday(&start,NULL);
	for (int i =39000; i < 49000; i++)								// 取后100张数字0来测试
	{
		Mat testData, testFeatures;
		Mat test = imread(dir + to_string(i) + ".jpg");

		Mat test_data;

		resize(test, test_data, Size(PSIZE, PSIZE));
		Mat imgGray;
		cvtColor(test_data, imgGray, COLOR_BGR2GRAY);
		Mat featureVector;
		ComputeLBPFeatureVector_Uniform(imgGray,Size(CSIZE,CSIZE),featureVector);

		int feature_dim = featureVector.cols;

		testFeatures = Mat::zeros(1, feature_dim, CV_32FC1);
		float * featurePtr = testFeatures.ptr<float>(0);
		for (int j = 0; j < feature_dim; j++) {
			*featurePtr = featureVector.at<float>(0,j);
			featurePtr++;
		}
		int k = model->predict(testFeatures);
		if (k == -1)
			count1++;
	}
	gettimeofday(&end,NULL);
	long timeuse=1000000*(end.tv_sec-start.tv_sec) + end.tv_usec -start.tv_usec;
	cout<<"用时："<<timeuse<<"微秒"<<endl;
double b1=double(count0)/double(5700);
double b2=double(count1)/double(10000);
cout<<count0<<"/5700"<<endl;
    cout << "检测正确比例：" << b1 << endl;
cout<<count1<<"/10000"<<endl;
		cout << "检测非人正确比例：" << b2<< endl;
	cout << "测试完毕." << endl;
	getchar();
	return 0;
}
