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

float computeRecall(Matrix<size_t> indice, vector<vector<size_t>> ground) {
	int count ;
	float sum = 0;
	vector<float> recall;
	for (size_t row = 0; row < indice.rows; row++) {
		count = 0;
		for (size_t col = 0; col < indice.cols; col++) {
			for (size_t g_col = 0; g_col < indice.cols; g_col++) {
				if (indice[row][col] == ground[row][g_col]) {
					count ++;
					break;
				}
			}
		}
		recall.push_back((1.0*count) / indice.cols);
	}
	for (int i = 0; i < recall.size(); i++) {
		sum += recall[i];
	}
	return sum/indice.rows ;
}

void buildAndSearch(Index<Distance> index, const Matrix<float>& queryall,  int checks, vector<vector<size_t>> groundTruthIndice) {
	ofstream outfile("kdtree_result.txt");
	clock_t starttime = clock();
	float recall;
	index.buildIndex();
	printf("build index done, %d ms\n", elapsedMilliseconds(starttime));
	int knn = 100;
	Matrix<size_t> indices(new size_t[queryall.rows*knn], queryall.rows, knn);
	Matrix<float> dists(new float[queryall.rows*knn], queryall.rows, knn);
	starttime = clock();
	int query_all_neighbor_count = index.knnSearch(queryall, indices, dists, knn, flann::SearchParams(checks));
	printf("search done, %d s\n",elapsedMilliseconds(starttime)/1000);
	recall = computeRecall(indices, groundTruthIndice);
	printf(" query recall for  %d nearest neighbor: %f\n", knn,recall);
	outfile << knn << " "<< elapsedMilliseconds(starttime) << " "<<recall << endl;
	outfile.close();
}

void main(int argc, char** argv)
{
	bool isSift = true;
	string directory_prefix = "../../";
	string base_filename = directory_prefix + (isSift ? "sift_base.fvecs" : "gist_base.fvecs");
	string query_filename = directory_prefix + (isSift ? "sift_query.fvecs" : "gist_query.fvecs");
	string groundtruth_filename = directory_prefix + (isSift ? "sift_groundtruth.ivecs" : "gist_groundtruth.ivecs");
	Matrix<float> dataset = TexmexDataSetReader::readFMatrix(base_filename);
	Matrix<float> queryall = TexmexDataSetReader::readFMatrix(query_filename);
	printf("data rows=%d, cols=%d\n", dataset.rows, dataset.cols);
	printf("query rows=%d, cols=%d\n", queryall.rows, queryall.cols);
	vector<vector<size_t>> groundTruthIndices = TexmexDataSetReader::readIvecs(groundtruth_filename);
	int tree_number = 30;
	int checks = 512;
	for(int i=0;i<=5;i++){
		tree_number += 2;
		//checks *= 2;
	Index<Distance> kdTreeIndex(dataset, KDTreeIndexParams(tree_number));
	printf("kdtree:%d tree, %d checks\n",tree_number,checks);
	buildAndSearch(kdTreeIndex, queryall,checks, groundTruthIndices);
	}
	//int table_number = 8;
	//int bucket_size = 50;
	//Index<Distance> lshIndex(dataset, LshIndexParams(table_number, bucket_size, 0));
	//printf("lsh:%d teble_number, %d bucket_size\n",table_number,bucket_size);
	//buildAndSearch(lshIndex, queryall,-1, groundTruthIndices);
	delete[] dataset.ptr();
	delete[] queryall.ptr();
	system("pause");
}
