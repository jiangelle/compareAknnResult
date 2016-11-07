#include <time.h>
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <fstream>

#include <stdio.h>
#include <vector>
#include <iostream>
#include "TexmexDataSetReader.hpp"
using namespace std;
using namespace flann;

typedef L2_Simple<float> Distance;
typedef Distance::ResultType DistanceType;
#define print_neighbor_count 10

int elapsedMilliseconds(clock_t startTime) {
	return 1000.0*(clock() - startTime) / CLOCKS_PER_SEC;
}

float computePrecision(Matrix<size_t> indice, vector<vector<size_t>> ground) {
	int count = 0;
	bool flag = false;
	for (size_t row = 0; row < indice.rows; row++) {
		for (size_t col = 0; col < indice.cols; col++) {
			if (indice[row][col] == ground[row][0]) {
				flag = true;
				break;
			}
			else {
				flag = false;
			}
		}
		if (flag == true) {
			count ++ ;
		}
	}
	return (count*1.0)/indice.rows ;
}

float computeMAP(Matrix<size_t> indice, vector<vector<size_t>> ground) {
	vector<float> queryMAP;
	float sumMAP = 0;
	float averageMAP;
	float sum = 0.0;
	for (int i = 0; i < indice.rows; i++) {
		averageMAP = 0;
		sum = 0;
		for (int j = 1; j <= indice.cols; j++) {
			for (int k = 1; k <= ground[0].size(); k++) {
				if (indice[i][j] == ground[i][k]) {
					sum = sum + (1.0* j)/k;
					break;
				}
				else {
					sum += 0;
				}
			}
		}
		averageMAP = sum / indice.cols;
		queryMAP.push_back(averageMAP);
	}
	for (int i = 0; i < queryMAP.size(); i++) {
		sumMAP += queryMAP[i];
	}
	return sumMAP / queryMAP.size();
}

void buildAndSearch(Index<Distance> index, const Matrix<float>& queryall,  int checks, vector<vector<size_t>> groundTruthIndice) {
	clock_t starttime = clock();
	index.buildIndex();
	printf("build index done, %d ms\n", elapsedMilliseconds(starttime));
	int knn = 100;
	//for(int knn = 1;knn<=1000;knn+= 50){
	Matrix<size_t> indices(new size_t[queryall.rows*knn], queryall.rows, knn);
	Matrix<float> dists(new float[queryall.rows*knn], queryall.rows, knn);
	starttime = clock();
	int query_all_neighbor_count = index.knnSearch(queryall, indices, dists, knn, flann::SearchParams(checks));
	printf("search  %d done, %d ms\n",knn, elapsedMilliseconds(starttime));
	printf(" query recall for nearest neighbor: %f\n", computePrecision(indices, groundTruthIndice));
	printf("MAP: %f\n", computeMAP(indices, groundTruthIndice));
	//}
}

void main(int argc, char** argv)
{
	bool isSift = true;
	clock_t starttime = clock();
	string directory_prefix = "../../";
	string base_filename = directory_prefix + (isSift ? "sift_base.fvecs" : "gist_base.fvecs");
	string query_filename = directory_prefix + (isSift ? "sift_query.fvecs" : "gist_query.fvecs");
	string learn_filename = directory_prefix + (isSift ? "sift_learn.fvecs" : "gist_learn.fvecs");
	string groundtruth_filename = directory_prefix + (isSift ? "sift_groundtruth.ivecs" : "gist_groundtruth.ivecs");
	Matrix<float> dataset = TexmexDataSetReader::readFMatrix(base_filename);
	Matrix<float> queryall = TexmexDataSetReader::readFMatrix(query_filename);
	Matrix<float> learndata = TexmexDataSetReader::readFMatrix(learn_filename);
	printf("read data done, %d ms\n", elapsedMilliseconds(starttime));
	printf("data rows=%d, cols=%d\n", dataset.rows, dataset.cols);
	printf("query rows=%d, cols=%d\n", queryall.rows, queryall.cols);
	printf("learn rows=%d, cols=%d\n", learndata.rows, learndata.cols);
	vector<vector<size_t>> groundTruthIndices = TexmexDataSetReader::readIvecs(groundtruth_filename);
	Index<Distance> kdTreeIndex(dataset, KDTreeIndexParams(15));
	printf("kdtree:\n");
	buildAndSearch(kdTreeIndex, queryall,256, groundTruthIndices);
	Index<Distance> lshIndex(dataset, LshIndexParams(8, 50, 0));
	ofstream outfile;
	outfile.open("kdtree_result.txt");
	outfile << "kdtree\n" << endl;
	outfile.close();
	printf("lsh:\n");
	buildAndSearch(lshIndex, queryall,  -1, groundTruthIndices);
	delete[] dataset.ptr();
	delete[] queryall.ptr();
	system("pause");
}
