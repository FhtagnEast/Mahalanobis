// Mahalanobis1.cpp: определяет точку входа для консольного приложения.

#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <tchar.h>
#include <math.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>



using namespace std;
using namespace cv;

class PixelMah
{
public:
	int size = 3;
	float B;
	float G;
	float R;
	//float H;
	//float S;
	//float V;
	PixelMah()
	{	
		size = 3;
		B = 0;
		G = 0;
		R = 0;
		//H = 0;
		//S = 0;
		//V = 0;
	}

	float out_param(int num)
	{
		if (num == 0) return B;
		if (num == 1) return G;
		if (num == 2) return R;
		//if (num == 3) return H;
		//if (num == 4) return S;
		//if (num == 5) return V;
	}

	cv::Mat vectorize()
	{
		cv::Mat mean_0(1, size, CV_32F);
		mean_0.at<float>(0, 0) = B;
		mean_0.at<float>(0, 1) = G;
		mean_0.at<float>(0, 2) = R;
		//mean_0.at<float>(0, 3) = H;
		//mean_0.at<float>(0, 4) = S;
		//mean_0.at<float>(0, 5) = V;
		return mean_0;
	}
};


void vecmatwrite(const string& filename, const vector<Mat>& matrices)
{
	ofstream fs(filename, fstream::binary);

	for (size_t i = 0; i < matrices.size(); ++i)
	{
		const Mat& mat = matrices[i];

		// Header
		int type = mat.type();
		int channels = mat.channels();
		fs.write((char*)&mat.rows, sizeof(int));    // rows
		fs.write((char*)&mat.cols, sizeof(int));    // cols
		fs.write((char*)&type, sizeof(int));        // type
		fs.write((char*)&channels, sizeof(int));    // channels

													// Data
		if (mat.isContinuous())
		{
			fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
		}
		else
		{
			int rowsz = CV_ELEM_SIZE(type) * mat.cols;
			for (int r = 0; r < mat.rows; ++r)
			{
				fs.write(mat.ptr<char>(r), rowsz);
			}
		}
	}
}

vector<Mat> vecmatread(const string& filename)
{
	vector<Mat> matrices;
	ifstream fs(filename, fstream::binary);

	// Get length of file
	fs.seekg(0, fs.end);
	int length = fs.tellg();
	fs.seekg(0, fs.beg);

	while (fs.tellg() < length)
	{
		// Header
		int rows, cols, type, channels;
		fs.read((char*)&rows, sizeof(int));         // rows
		fs.read((char*)&cols, sizeof(int));         // cols
		fs.read((char*)&type, sizeof(int));         // type
		fs.read((char*)&channels, sizeof(int));     // channels

													// Data
		Mat mat(rows, cols, type);
		fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

		matrices.push_back(mat);
	}
	return matrices;
}

vector<cv::Mat> learning(int blocksize);
void testing(int blocksize, cv::Mat &rgb_img, vector<cv::Mat> covariations);

vector<PixelMah> getVectorOfMeanBlockValues(int blocksize, vector<Point> class_N_coord);

PixelMah getClassMean(vector<PixelMah> vector);


Mat calcCovariationMatrix(vector<PixelMah> vector, PixelMah mean);

cv::Mat img_rgb;
cv::Mat img_hsv;

int main() 
{
	int blocksize;
	cout << "Input size of block ";
	cin >> blocksize;
	img_rgb = cv::imread("C:\\pic\\3.jpg", CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(img_rgb, img_hsv, CV_BGR2HSV);
	//vector<cv::Mat> covAndMeans = learning(blocksize);// обучение
	//vecmatwrite("testtulips.bin", covAndMeans); //обучение
	vector<Mat> covAndMeans = vecmatread("testtulips.bin"); //классификация 
	
	cout << covAndMeans[0]<< endl;
	cout << covAndMeans[1] << endl;
	cout << covAndMeans[2] << endl;
	cout << covAndMeans[3] << endl;
	cout << covAndMeans[0].inv() << endl << covAndMeans[2].inv() << endl;
	testing(blocksize,img_rgb, covAndMeans); //классификация
	int dummy;
	cin >> dummy;
	return 0;
}

vector<PixelMah> coords2pixels(int blocksize, cv::Point leftcorner)
{
	vector<PixelMah> output;
	for (int i = 0; i < blocksize; i++)
	{
		for (int j = 0; j < blocksize; j++)
		{
			PixelMah pix;
			int x = j + leftcorner.x;
			int y = i + leftcorner.y;
			float R = img_rgb.at<cv::Vec3b>(y, x)[2];
			float G = img_rgb.at<cv::Vec3b>(y, x)[1];
			float B = img_rgb.at<cv::Vec3b>(y, x)[0];
			pix.B = B / (B + G + R);
			pix.G = G / (B + G + R);
			pix.R = R / (B + G + R);
			//pix.H = img_hsv.at<cv::Vec3b>(y, x)[0];
			//pix.S = img_hsv.at<cv::Vec3b>(y, x)[1];
			//pix.V = img_hsv.at<cv::Vec3b>(y, x)[2];
			output.push_back(pix);
		}
	}
	return output;
}

vector<cv::Mat> learning(int blocksize)
{
	int x = 0;
	int y = 0;
	int xMax = img_rgb.cols;
	int yMax = img_rgb.rows;
	vector<cv::Point> class_0_coord;
	vector<cv::Point> class_1_coord;

	while (1)
	{
		cout << "Enter X coordinate of first class, to exit enter more than your image's maximum x coordinate" << endl;
		cin >> x;
		cout << "Enter Y coordinate of first class" << endl;
		cin >> y;
		if (x < xMax-blocksize && y < yMax-blocksize)
		{
			cv::Point pix_coords;
			pix_coords.x = x;
			pix_coords.y = y;
			class_0_coord.push_back(pix_coords);
		}
		else
		{
			break;
		}
	}

	while (1)
	{
		cout << "Enter X coordinate of second class, to exit enter more than your image's maximum x coordinate" << endl;
		cin >> x;
		cout << "Enter Y coordinate of second class" << endl;
		cin >> y;
		if (x < xMax-blocksize && y < yMax-blocksize)
		{
			cv::Point pix_coords;
			pix_coords.x = x;
			pix_coords.y = y;
			class_1_coord.push_back(pix_coords);
		}
		else
		{
			break;
		}
	}

	vector<PixelMah> vector_0_ofMeanBlockValues = getVectorOfMeanBlockValues(blocksize, class_0_coord);
	vector<PixelMah> vector_1_ofMeanBlockValues = getVectorOfMeanBlockValues(blocksize, class_1_coord);

	PixelMah class_0_Mean = getClassMean(vector_0_ofMeanBlockValues);
	PixelMah class_1_Mean = getClassMean(vector_1_ofMeanBlockValues);

	Mat class_0_CovMatrix = calcCovariationMatrix(vector_0_ofMeanBlockValues, class_0_Mean);
	Mat class_1_CovMatrix = calcCovariationMatrix(vector_1_ofMeanBlockValues, class_1_Mean);

	vector<Mat> output;
	output.push_back(class_0_CovMatrix);
	output.push_back(class_0_Mean.vectorize());
	output.push_back(class_1_CovMatrix);
	output.push_back(class_1_Mean.vectorize());
	return output;
}

vector<PixelMah> getVectorOfMeanBlockValues(int blocksize, vector<Point> class_N_coord)
{
	vector<PixelMah> output;

	for (int i = 0; i < (class_N_coord.size()); i++)
	{
		cv::Point current_coord = class_N_coord[i];
		vector<PixelMah> blocks = coords2pixels(blocksize, class_N_coord[i]);
		PixelMah meanOfBlockPixel;
		for (int j = 0; j < blocks.size(); j++)
		{
			meanOfBlockPixel.B += blocks[j].B;
			meanOfBlockPixel.G += blocks[j].G;
			meanOfBlockPixel.R += blocks[j].R;
			//meanOfBlockPixel.H += blocks[j].H;
			//meanOfBlockPixel.S += blocks[j].S;
			//meanOfBlockPixel.V += blocks[j].V;
		}
		meanOfBlockPixel.B = meanOfBlockPixel.B / blocks.size();
		meanOfBlockPixel.G = meanOfBlockPixel.G / blocks.size();
		meanOfBlockPixel.R = meanOfBlockPixel.R / blocks.size();
		//meanOfBlockPixel.H = meanOfBlockPixel.H / blocks.size();
		//meanOfBlockPixel.S = meanOfBlockPixel.S / blocks.size();
		//meanOfBlockPixel.V = meanOfBlockPixel.V / blocks.size();
		output.push_back(meanOfBlockPixel);
	}
	return output;
}

PixelMah getClassMean(vector<PixelMah> vector)
{
	PixelMah accumulator;
	for (int i = 0; i < (vector.size()); i++)
	{
		accumulator.R += vector[i].R;
		accumulator.G += vector[i].G;
		accumulator.B += vector[i].B;
		//accumulator.H += vector[i].H;
		//accumulator.S += vector[i].S;
		//accumulator.V += vector[i].V;
	}
	accumulator.R = accumulator.R / (vector.size());
	accumulator.G = accumulator.G / (vector.size());
	accumulator.B = accumulator.B / (vector.size());
	//accumulator.H = accumulator.H / (vector.size());
	//accumulator.S = accumulator.S / (vector.size());
	//accumulator.V = accumulator.V / (vector.size());
	return accumulator;
}

Mat calcCovariationMatrix(vector<PixelMah> vector, PixelMah mean)
{
	Mat output(mean.size, mean.size, CV_32F);
	for (int i = 0; i < mean.size; i++)
	{
		for (int j = 0; j < mean.size; j++)
		{
			float nominator = 0;
			for (int k = 0; k<vector.size(); k++)
			{
				nominator += (vector[k].out_param(i) - mean.out_param(i))
					* (vector[k].out_param(j) - mean.out_param(j));
			}
			output.at<float>(i, j) = nominator / (vector.size());
		}
	}
	return output;
}


void testing(int blocksize, cv::Mat &rgb_img, vector<cv::Mat> covariations)
{
	cv::Mat hsv_img = rgb_img.clone();
	cv::cvtColor(rgb_img, hsv_img, CV_BGR2HSV);
	int block_rows = rgb_img.rows / blocksize;
	int block_cols = rgb_img.cols / blocksize;
	vector<vector<cv::Mat>> blocks_mean;
	for (int i = 0; i < block_rows; i++)
	{
		vector<cv::Mat> row;
		for (int j = 0; j < block_cols; j++)
		{
			cv::Point leftcorner;
			leftcorner.x = j*blocksize;
			leftcorner.y = i*blocksize;
			PixelMah meanOfBlockPixel;
			vector<PixelMah> blockvector;
			blockvector = coords2pixels(blocksize, leftcorner);
			for (int j = 0; j < blockvector.size(); j++)
			{
				meanOfBlockPixel.B += blockvector[j].B;
				meanOfBlockPixel.G += blockvector[j].G;
				meanOfBlockPixel.R += blockvector[j].R;
				//meanOfBlockPixel.H += blockvector[j].H;
				//meanOfBlockPixel.S += blockvector[j].S;
				//meanOfBlockPixel.V += blockvector[j].V;
			}
			meanOfBlockPixel.B = meanOfBlockPixel.B / blockvector.size();
			meanOfBlockPixel.G = meanOfBlockPixel.G / blockvector.size();
			meanOfBlockPixel.R = meanOfBlockPixel.R / blockvector.size();
			//meanOfBlockPixel.H = meanOfBlockPixel.H / blockvector.size();
			//meanOfBlockPixel.S = meanOfBlockPixel.S / blockvector.size();
			//meanOfBlockPixel.V = meanOfBlockPixel.V / blockvector.size();
			cv::Mat block(1, meanOfBlockPixel.size, CV_32F);
			block = meanOfBlockPixel.vectorize();
			row.push_back(block);
		}
		blocks_mean.push_back(row);
	}
	cv::Mat out_class(block_rows, block_cols, CV_32F);
	PixelMah test;
	for (int i = 0; i < block_rows; i++)
	{
		for (int j = 0; j < block_cols; j++)
		{	
			cv::Mat dist_0(1, test.size, CV_32F);
			cv::Mat dist_1(1, test.size, CV_32F);
			dist_0 = (blocks_mean[i][j] - covariations[1]);
			cv::Mat dist_0t(test.size, 1, CV_32F);
			dist_0t = dist_0.t();
			dist_1 = (blocks_mean[i][j] - covariations[3]);
			cv::Mat dist_1t(test.size, 1, CV_32F);
			cv::Mat invmat0(test.size, test.size, CV_32F);
			cv::Mat invmat1(test.size, test.size, CV_32F);
			invmat0 = covariations[0].inv();
			invmat1 = covariations[2].inv();
			dist_1t = dist_1.t();
			cv::Mat dist_0_fin0 = dist_0*invmat0;
			cv::Mat dist_1_fin0 = dist_1*invmat1;
			cv::Mat dist_0_fin = dist_0_fin0*dist_0t;
			cv::Mat dist_1_fin = dist_1_fin0*dist_1t;
			cv::Mat blck = blocks_mean[i][j];
			cv::Mat test1;
			float dist_0_val = sqrt(dist_0_fin.at<float>(0, 0));
			float dist_1_val = sqrt(dist_1_fin.at<float>(0, 0));
			if (dist_0_val <= dist_1_val)
			{
				out_class.at<float>(i, j) = 255;
			}
			else
			{
				out_class.at<float>(i, j) = 0;
			}
		}

	}
	imshow("Img", out_class);
	cvWaitKey(0);
}