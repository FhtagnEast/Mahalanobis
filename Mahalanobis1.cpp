//// Mahalanobis1.cpp: определяет точку входа для консольного приложения.
////
//
//#include "stdafx.h"
//#include <math.h>
//#include <iostream>
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <fstream>
//using namespace std;
//using namespace cv;
////test
//void vecmatwrite(const string& filename, const vector<Mat>& matrices)
//{
//	ofstream fs(filename, fstream::binary);
//
//	for (size_t i = 0; i < matrices.size(); ++i)
//	{
//		const Mat& mat = matrices[i];
//
//		// Header
//		int type = mat.type();
//		int channels = mat.channels();
//		fs.write((char*)&mat.rows, sizeof(int));    // rows
//		fs.write((char*)&mat.cols, sizeof(int));    // cols
//		fs.write((char*)&type, sizeof(int));        // type
//		fs.write((char*)&channels, sizeof(int));    // channels
//
//													// Data
//		if (mat.isContinuous())
//		{
//			fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
//		}
//		else
//		{
//			int rowsz = CV_ELEM_SIZE(type) * mat.cols;
//			for (int r = 0; r < mat.rows; ++r)
//			{
//				fs.write(mat.ptr<char>(r), rowsz);
//			}
//		}
//	}
//}
//
//vector<Mat> vecmatread(const string& filename)
//{
//	vector<Mat> matrices;
//	ifstream fs(filename, fstream::binary);
//
//	// Get length of file
//	fs.seekg(0, fs.end);
//	int length = fs.tellg();
//	fs.seekg(0, fs.beg);
//
//	while (fs.tellg() < length)
//	{
//		// Header
//		int rows, cols, type, channels;
//		fs.read((char*)&rows, sizeof(int));         // rows
//		fs.read((char*)&cols, sizeof(int));         // cols
//		fs.read((char*)&type, sizeof(int));         // type
//		fs.read((char*)&channels, sizeof(int));     // channels
//
//													// Data
//		Mat mat(rows, cols, type);
//		fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);
//
//		matrices.push_back(mat);
//	}
//	return matrices;
//}
//
//void printMatrix(cv::Mat M) {
//	cout << "Matrix:" << M.type() << endl;
//	// dont print empty matrices
//	if (M.empty()) {
//		cout << "---" << endl;
//		return;
//	}
//	// loop through columns and rows of the matrix
//	for (int i = 0; i < M.rows; i++) {
//		for (int j = 0; j < M.cols; j++) {
//			cout << M.at<float>(i, j) << ", " << endl;
//		}
//		cout << "line\n" << endl;
//	}
//}
//
//void printMatrixint(cv::Mat M) {
//	cout << "Matrix:" << M.type() << endl;
//	// dont print empty matrices
//	if (M.empty()) {
//		cout << "---" << endl;
//		return;
//	}
//	// loop through columns and rows of the matrix
//	for (int i = 0; i < M.rows; i++) {
//		for (int j = 0; j < M.cols; j++) {
//			cout << M.at<int>(i, j) << ", " << endl;
//		}
//		cout << "Line\n" << endl;
//	}
//}
//
/////test
//vector<cv::Mat> classifier_learning(int blocksize, cv::Mat &rgb_img);
//void classifying(int blocksize, cv::Mat &rgb_img, vector<cv::Mat> covariations);
//
//class block_value
//{
//public:
//	float B;
//	float G;
//	float R;
//	float H;
//	float S;
//	float V;
//	block_value()
//	{
//		B = 0;
//		G = 0;
//		R = 0;
//		H = 0;
//		S = 0;
//		V = 0;
//	}
//
//	float out_param(int num)
//	{
//		if (num == 0) return B;
//		if (num == 1) return G;
//		if (num == 2) return R;
//		if (num == 3) return H;
//		if (num == 4) return S;
//		if (num == 5) return V;
//	}
//
//	cv::Mat vectorize()
//	{
//		cv::Mat mean_0(1, 6, CV_32F);
//		mean_0.at<float>(0, 0) = B;
//		mean_0.at<float>(0, 1) = G;
//		mean_0.at<float>(0, 2) = R;
//		mean_0.at<float>(0, 3) = H;
//		mean_0.at<float>(0, 4) = S;
//		mean_0.at<float>(0, 5) = V;
//		return mean_0;
//	}
//};
//
//vector<block_value> coord2blockvector(int blocksize, cv::Point leftcorner, cv::Mat &img, cv::Mat &hsv_img);
//
//int main()
//{
//	int blocksize;
//	cout << "Input size of block ";
//	cin >> blocksize;
//	cv::Mat img;
//	//img = cv::imread("C:\\pic\\Tulips.jpg", CV_LOAD_IMAGE_COLOR);
//	img = cv::imread("C:\\pic\\Penguins.jpg", CV_LOAD_IMAGE_COLOR);
//	//img = cv::imread("C:\\Tulips\\2.jpg", CV_LOAD_IMAGE_COLOR);
//	vector<cv::Mat> matrices=classifier_learning(blocksize, img);
//	vector<Mat> matrices=vecmatread("testpin.bin");
//	//vecmatwrite("testpin.bin", matrices);
//	classifying(blocksize, img, matrices);
////	cv::Mat img1 = cv::imread("C:\\pic\\rusflg.jpg", CV_LOAD_IMAGE_COLOR);
////	classifying(blocksize,img1,matrices);
//	int dummy;
//	cin >> dummy;
//	return 0;
//	
//}
//
///*
//vector<cv::Vec3b> block2vector(int blocksize, cv::Point leftcorner, cv::Mat &rgb_img, cv::Mat &hsv_img)
//{
//	vector<cv::Vec3b> row;
//	for (int i = 0; i < blocksize; i++)
//	{
//		for(int j = 0; j < blocksize; j++)
//		{
//			int x = j + leftcorner.x;
//			int y = i + leftcorner.y;
//			cv::Vec3b pix = img.at<cv::Vec3b>(y, x);
//			row.push_back(pix);
//		}
//
//	}
//	return (row);
//}
//*/
//
//vector<block_value> coord2blockvector(int blocksize, cv::Point leftcorner, cv::Mat &rgb_img, cv::Mat &hsv_img)
//{
//	vector<block_value> row;
//	for (int i = 0; i < blocksize; i++)
//	{
//		for (int j = 0; j < blocksize; j++)
//		{
//			block_value pix;
//			int x = j + leftcorner.x;
//			int y = i + leftcorner.y;
//			float R = rgb_img.at<cv::Vec3b>(y, x)[2];
//			float G = rgb_img.at<cv::Vec3b>(y, x)[1];
//			float B = rgb_img.at<cv::Vec3b>(y, x)[0];
//			pix.B = B / (B + G + R);
//			pix.G = G / (B + G + R);
//			pix.R = R / (B + G + R);
//			pix.H = hsv_img.at<cv::Vec3b>(y, x)[0];
//			pix.S = hsv_img.at<cv::Vec3b>(y, x)[1];
//			pix.V = hsv_img.at<cv::Vec3b>(y, x)[2];
//			row.push_back(pix);
//		}
//
//	}
//	return (row);
//}
//
//vector<cv::Mat> classifier_learning(int blocksize, cv::Mat &img)
//{
//	cv::Mat hsv_img = img.clone();
//	cv::cvtColor(img,hsv_img,CV_BGR2HSV);
//	int entered_value = 0;
//	int X, Y;
//	X = img.cols;
//	Y = img.rows;
//	vector<cv::Point> class_0_coord;
//	vector<cv::Point> class_1_coord;
//	vector<block_value> blocks_0_value;
//	vector<block_value> blocks_1_value;
//	while (entered_value <= X)
//	{
//		int entered_value_1;
//		cv::Point pix_coords;
//		cout << "Enter X coordinate of first class, to exit enter more than your image's maximum x coordinate" << endl;
//		cin >> entered_value;
//		cout << "Enter Y coordinate of first class" << endl;
//		cin >> entered_value_1;
//		if ((entered_value+blocksize <= X) & (entered_value_1+blocksize <= Y))
//		{
//			pix_coords.x = entered_value;
//			pix_coords.y = entered_value_1;
//			class_0_coord.push_back(pix_coords);
//		}
//	}
//	entered_value = 0;
//	while (entered_value <= X)
//	{
//		int entered_value_1;
//		cv::Point pix_coords;
//		cout << "Enter X coordinate of second class, to exit enter more than your image's maximum x coordinate" << endl;
//		cin >> entered_value;
//		cout << "Enter Y coordinate of second class" << endl;
//		cin >> entered_value_1;
//		if ((entered_value <= X) & (entered_value_1 <= Y))
//		{
//			pix_coords.x = entered_value;
//			pix_coords.y = entered_value_1;
//			class_1_coord.push_back(pix_coords);
//		}
//	}
//
//	vector<block_value> pixels_mean_param_class0;
//	for (int i = 0; i < (class_0_coord.size()); i ++)
//	{
//		cv::Point current_coord = class_0_coord[i];
//		block_value block_accumulator;
//		vector<block_value> blockvector;
//		blockvector=coord2blockvector(blocksize, current_coord, img, hsv_img);
//		for (int j = 0; j < blockvector.size(); j++)
//		{
//			block_accumulator.B += blockvector[j].B;
//			block_accumulator.G += blockvector[j].G;
//			block_accumulator.R += blockvector[j].R;
//			block_accumulator.H += blockvector[j].H;
//			block_accumulator.S += blockvector[j].S;
//			block_accumulator.V += blockvector[j].V;
//		}
//		block_accumulator.B = block_accumulator.B / blockvector.size();
//		block_accumulator.G = block_accumulator.G / blockvector.size();
//		block_accumulator.R = block_accumulator.R / blockvector.size();
//		block_accumulator.H = block_accumulator.H / blockvector.size();
//		block_accumulator.S = block_accumulator.S / blockvector.size();
//		block_accumulator.V = block_accumulator.V / blockvector.size();
//		pixels_mean_param_class0.push_back(block_accumulator);		
//	}
//
//	vector<block_value> pixels_mean_param_class1;
//	for (int i = 0; i < class_1_coord.size(); i++)
//	{
//		cv::Point current_coord = class_1_coord[i];
//		block_value block_accumulator;
//		vector<block_value> blockvector;
//		blockvector = coord2blockvector(blocksize, current_coord, img, hsv_img);
//		for (int j = 0; j < blockvector.size(); j++)
//		{
//			block_accumulator.B += blockvector[j].B;
//			block_accumulator.G += blockvector[j].G;
//			block_accumulator.R += blockvector[j].R;
//			block_accumulator.H += blockvector[j].H;
//			block_accumulator.S += blockvector[j].S;
//			block_accumulator.V += blockvector[j].V;
//		}
//		block_accumulator.B = block_accumulator.B / blockvector.size();
//		block_accumulator.G = block_accumulator.G / blockvector.size();
//		block_accumulator.R = block_accumulator.R / blockvector.size();
//		block_accumulator.H = block_accumulator.H / blockvector.size();
//		block_accumulator.S = block_accumulator.S / blockvector.size();
//		block_accumulator.V = block_accumulator.V / blockvector.size();
//		pixels_mean_param_class1.push_back(block_accumulator);
//	}
//	
//	block_value class_0_mean;
//	block_value class_1_mean;
//	float class_0_mean_acc_R = 0;
//	float class_0_mean_acc_G = 0;
//	float class_0_mean_acc_B = 0;
//	float class_0_mean_acc_H = 0;
//	float class_0_mean_acc_S = 0;
//	float class_0_mean_acc_V = 0;
//	float class_1_mean_acc_R = 0;
//	float class_1_mean_acc_G = 0;
//	float class_1_mean_acc_B = 0;
//	float class_1_mean_acc_H = 0;
//	float class_1_mean_acc_S = 0;
//	float class_1_mean_acc_V = 0;
//	for (int i = 0; i < (class_0_coord.size()); i++)
//	{
//		class_0_mean_acc_R += pixels_mean_param_class0[i].R;
//		class_0_mean_acc_G += pixels_mean_param_class0[i].G;
//		class_0_mean_acc_B += pixels_mean_param_class0[i].B;
//		class_0_mean_acc_H += pixels_mean_param_class0[i].H;
//		class_0_mean_acc_S += pixels_mean_param_class0[i].S;
//		class_0_mean_acc_V += pixels_mean_param_class0[i].V;
//	}
//	for (int i = 0; i < (class_1_coord.size()); i++)
//	{
//		class_1_mean_acc_R += pixels_mean_param_class1[i].R;
//		class_1_mean_acc_G += pixels_mean_param_class1[i].G;
//		class_1_mean_acc_B += pixels_mean_param_class1[i].B;
//		class_1_mean_acc_H += pixels_mean_param_class1[i].H;
//		class_1_mean_acc_S += pixels_mean_param_class1[i].S;
//		class_1_mean_acc_V += pixels_mean_param_class1[i].V;
//	}
//	class_0_mean.R = class_0_mean_acc_R / (class_0_coord.size());
//	class_0_mean.G = class_0_mean_acc_G / (class_0_coord.size());
//	class_0_mean.B = class_0_mean_acc_B / (class_0_coord.size());
//	class_0_mean.H = class_0_mean_acc_H / (class_0_coord.size());
//	class_0_mean.S = class_0_mean_acc_S / (class_0_coord.size());
//	class_0_mean.V = class_0_mean_acc_V / (class_0_coord.size());
//	class_1_mean.R = class_1_mean_acc_R / (class_1_coord.size());
//	class_1_mean.G = class_1_mean_acc_G / (class_1_coord.size());
//	class_1_mean.B = class_1_mean_acc_B / (class_1_coord.size());
//	class_1_mean.H = class_1_mean_acc_H / (class_1_coord.size());
//	class_1_mean.S = class_1_mean_acc_S / (class_1_coord.size());
//	class_1_mean.V = class_1_mean_acc_V / (class_1_coord.size());
//
//	cv::Mat cov_mat_0(6,6,CV_32F);
//	for (int i = 0; i < 6; i++)
//	{
//		for (int j = 0; j < 6; j++)
//		{
//			double param_acc = 0;
//			float request_px_0 = 0;
//			float request_px_1 = 0;
//			float current_mean_0 = class_0_mean.out_param(i);
//			float current_mean_1 = class_0_mean.out_param(j);
//			for (int k = 0; k < (class_0_coord.size()); k++)
//			{
//				request_px_0 = pixels_mean_param_class0[k].out_param(i);
//				request_px_1 = pixels_mean_param_class0[k].out_param(j);
//				param_acc += (request_px_0 - current_mean_0)*(request_px_1 - current_mean_1);
//			}
//			cov_mat_0.at<float>(i, j) = param_acc / (class_0_coord.size());
//		}
//	}
//	cv::Mat cov_mat_1(6, 6, CV_32F);
//	cout << cov_mat_0 << endl;
//	for (int i = 0; i < 6; i++)
//	{
//		for (int j = 0; j < 6; j++)
//		{
//			double param_acc = 0;
//			float request_px_0 = 0;
//			float request_px_1 = 0;
//			float current_mean_0 = class_0_mean.out_param(i);
//			float current_mean_1 = class_0_mean.out_param(j);
//			for (int k = 0; k < (class_1_coord.size()); k++)
//			{
//				request_px_0 = pixels_mean_param_class1[k].out_param(i);
//				request_px_1 = pixels_mean_param_class1[k].out_param(j);
//				param_acc += (request_px_0 - current_mean_0)*(request_px_1 - current_mean_1);
//			}
//			cov_mat_1.at<float>(i, j) = param_acc / (class_1_coord.size());
//		}
//	}
//				
//	cv::Mat mean_0(1, 6, CV_32F);
//	mean_0 = class_0_mean.vectorize();
//	cv::Mat mean_1(1, 6, CV_32F);
//	mean_1 = class_1_mean.vectorize();
//	//cout << cov_mat_0 << endl << endl << cov_mat_1 << endl << endl << mean_0 << endl << endl << mean_1<<endl<<endl;	
//	vector<cv::Mat> output;
//	output.push_back(cov_mat_0);
//	output.push_back(mean_0);
//	output.push_back(cov_mat_1);
//	output.push_back(mean_1);
//	return output;
//}
//
//void classifying(int blocksize, cv::Mat &rgb_img, vector<cv::Mat> covariations)
//{
//	cv::Mat hsv_img = rgb_img.clone();
//	cv::cvtColor(rgb_img, hsv_img, CV_BGR2HSV); 
//	int block_rows = rgb_img.rows / blocksize;
//	int block_cols = rgb_img.cols / blocksize;
//	vector<vector<cv::Mat>> blocks_mean;
//	for (int i = 0; i < block_rows; i++)
//	{
//		vector<cv::Mat> row;
//		for (int j = 0; j < block_cols; j++)
//		{
//			cv::Point leftcorner;
//			leftcorner.x = j*blocksize;
//			leftcorner.y = i*blocksize;
//			block_value block_accumulator;
//			vector<block_value> blockvector;
//			blockvector = coord2blockvector(blocksize, leftcorner, rgb_img, hsv_img);
//			for (int j = 0; j < blockvector.size(); j++)
//			{
//				block_accumulator.B += blockvector[j].B;
//				block_accumulator.G += blockvector[j].G;
//				block_accumulator.R += blockvector[j].R;
//				block_accumulator.H += blockvector[j].H;
//				block_accumulator.S += blockvector[j].S;
//				block_accumulator.V += blockvector[j].V;
//			}
//			block_accumulator.B = block_accumulator.B / blockvector.size();
//			block_accumulator.G = block_accumulator.G / blockvector.size();
//			block_accumulator.R = block_accumulator.R / blockvector.size();
//			block_accumulator.H = block_accumulator.H / blockvector.size();
//			block_accumulator.S = block_accumulator.S / blockvector.size();
//			block_accumulator.V = block_accumulator.V / blockvector.size();
//			cv::Mat block(1, 6, CV_32F);
//			block = block_accumulator.vectorize();
//			row.push_back(block);
//		}
//		blocks_mean.push_back(row);
//	}
//	cv::Mat out_class(block_rows, block_cols, CV_32F);
// cout << covariations[0] * covariations[0].inv(DECOMP_LU) << endl << covariations[2] * covariations[2].inv(DECOMP_LU) << endl << covariations[0] << endl << covariations[2] << endl << covariations[0].inv(DECOMP_LU) << endl << covariations[2].inv(DECOMP_LU) << endl;
//	for (int i = 0; i < block_rows; i++)
//	{
//		for (int j = 0; j < block_cols; j++)
//		{
//			cv::Mat dist_0(1, 6, CV_32F);
//			cv::Mat dist_1(1, 6, CV_32F);
//			dist_0 = (blocks_mean[i][j] - covariations[1]);
//			cv::Mat dist_0t(6, 1, CV_32F);
//			cv::Mat weight(1, 6, CV_32F);
//			weight = (Mat_<float>(1,6)<<1,1,1,1,1,1);
//			dist_0 = dist_0.mul(weight);
//			dist_0t=dist_0.t();
//			dist_1 = (blocks_mean[i][j] - covariations[3]);
//			cv::Mat dist_1t(6, 1, CV_32F);
//			cv::Mat invmat0(6, 6, CV_32F);
//			cv::Mat invmat1(6, 6, CV_32F);
//			invmat0 = covariations[0].inv();
//			invmat1 = covariations[2].inv();
//			dist_1 = dist_1.mul(weight);
//			dist_1t = dist_1.t();
////			cout << dist_0 << endl << invmat0 << endl;
//			cv::Mat dist_0_fin0 = dist_0*invmat0;
//			cv::Mat dist_1_fin0 = dist_1*invmat1;
//			cv::Mat dist_0_fin = dist_0_fin0*dist_0t;
//			cv::Mat dist_1_fin = dist_1_fin0*dist_1t;
////			cv::Mat weight(1, 6, CV_32F);
//			weight = (Mat_<float>(1, 6) << 1, 1, 1, 1, 1, 1);
//			cv::Mat blck=blocks_mean[i][j];
//			blck = blck.mul(weight);
//			cv::Mat test1;
//			test1 = (Mat_<float>(6, 6) << 1, 0, 0, 0, 0, 0,
//				0, 1, 0, 0, 0, 0,
//				0, 0, 1, 0, 0, 0,
//				0, 0, 0, 1, 0, 0,
//				0, 0, 0, 0, 1, 0,
//				0, 0, 0, 0, 0, 1);
//			/*double k=Mahalanobis(blocks_mean[i][j],covariations[1],covariations[0]);
//			double k=Mahalanobis(blocks_mean[i][j],covariations[3],covariations[2]);*/
//		//	if (i==0) cout << covariations[0] * covariations[0].inv(DECOMP_LU) << endl << covariations[2] * covariations[2].inv(DECOMP_LU) << endl<<covariations[0]<<endl<<covariations[2]<<endl<< covariations[0].inv(DECOMP_LU) << endl << covariations[2].inv(DECOMP_LU)<<endl;
////			double k= Mahalanobis(blck, covariations[1], covariations[0].inv());
////			double k1= Mahalanobis(blck, covariations[3], covariations[2].inv());
//			float dist_0_val = sqrt(dist_0_fin.at<float>(0,0));
//	      	float dist_1_val = sqrt(dist_1_fin.at<float>(0,0));
//			cout << dist_0_val << " | " << dist_1_val << endl;
////			cout << k << " | " << k1 << endl;
////			if (k <= k1)
//			if(dist_0_val<=dist_1_val)
//			{
//				out_class.at<float>(i, j) = 255;
//			}
//			else
//			{
//				out_class.at<float>(i, j) = 0;
//			}
//		}
//
//	}
//	imshow("Img", out_class);
//	cvWaitKey(0);
//}