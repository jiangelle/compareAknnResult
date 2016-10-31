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

void printStatistics(const vector<vector<float>>& distancesVector) {
	vector<float> minDisVector;
	vector<float> maxDisVector;
	vector<float> meanDisVector;
	vector<float> stdVarDistanceVector;
	for (size_t neighbor_index = 0; neighbor_index < distancesVector[0].size(); ++neighbor_index) {
		minDisVector.push_back(FLT_MAX/10);
		maxDisVector.push_back(-FLT_MAX/10);
		meanDisVector.push_back(0);
		stdVarDistanceVector.push_back(0);
	}
	for (size_t neighbor_index = 0; neighbor_index < distancesVector[0].size(); ++neighbor_index) {
		for (size_t query_row = 0; query_row < distancesVector.size(); ++query_row) {
			if (neighbor_index < distancesVector[query_row].size()) {
				float distance = distancesVector[query_row][neighbor_index];
				minDisVector[neighbor_index] = min(minDisVector[neighbor_index], distance);
				maxDisVector[neighbor_index] = max(maxDisVector[neighbor_index], distance);
				meanDisVector[neighbor_index] += distance;
			}
		}
		meanDisVector[neighbor_index] /= distancesVector.size();
	}
	for (size_t neighbor_index = 0; neighbor_index < distancesVector[0].size(); ++neighbor_index) {
		for (size_t query_row = 0; query_row < distancesVector.size(); ++query_row) {
			float distance = distancesVector[query_row][neighbor_index];
			float diff = distance - meanDisVector[neighbor_index];
			stdVarDistanceVector[neighbor_index] += diff*diff;
		}
		stdVarDistanceVector[neighbor_index] /= distancesVector.size() - 1;
		stdVarDistanceVector[neighbor_index] = sqrt(stdVarDistanceVector[neighbor_index]);
	}

	for (size_t neighbor_index = 0; neighbor_index < print_neighbor_count; ++neighbor_index) {
		printf("neighbor_index=%d, min=%f, max=%f, mean=%f, stdVar=%f\n", neighbor_index, minDisVector[neighbor_index], maxDisVector[neighbor_index], meanDisVector[neighbor_index], stdVarDistanceVector[neighbor_index]);
	}
}

void printGroundTruthResult(const Matrix<float>& query, const Matrix<float>& baseMatrix, const vector<vector<size_t>>& groundTruthIndices) {
	vector<vector<float>> groundTruthDistances;
	Distance distance_functor;
	for (int query_row = 0; query_row < query.rows; query_row++) {
		vector<float> distances;
		float* query_feature = new float[query.cols];
		for (size_t i = 0; i < query.cols; ++i) {
			query_feature[i] = query[query_row][i];
		}
		for (int neighbor_index = 0; neighbor_index < groundTruthIndices[query_row].size(); neighbor_index++) {
			float* groundtruth_feature = new float[query.cols];
			for (size_t j = 0; j < query.cols; ++j) {
				groundtruth_feature[j] = baseMatrix[groundTruthIndices[query_row][neighbor_index]][j];
			}
			float distance = sqrt(distance_functor(query_feature, groundtruth_feature, query.cols));
			distances.push_back(distance);
			delete[] groundtruth_feature;
		}
		sort(distances.begin(), distances.end());
		delete[] query_feature;
		groundTruthDistances.push_back(distances);
	}
	printStatistics(groundTruthDistances);
}

void printResult(const flann::Matrix<float>& dists, const Matrix<size_t>& indices) {
	vector<vector<float>> floatDistances;
	for (int i = 0; i < dists.rows; i++) {
		floatDistances.push_back(vector<float>());
		for (int j = 0; j < dists.cols; j++) {
			if (indices[i][j] != -1) {
				floatDistances[i].push_back(sqrt(dists[i][j]));
			}
		}
		sort(floatDistances[i].begin(), floatDistances[i].end());
	}
	printStatistics(floatDistances);
}

void buildAndSearch(Index<Distance> index, const Matrix<float>& queryall, int knn, int checks) {
	Matrix<size_t> indices(new size_t[queryall.rows*knn], queryall.rows, knn);
	Matrix<float> dists(new float[queryall.rows*knn], queryall.rows, knn);
	memset(indices.ptr(), (size_t)-1, queryall.rows*knn*sizeof(size_t));
	clock_t starttime = clock();
	index.buildIndex();
	printf("build index done, %d ms\n", elapsedMilliseconds(starttime));
	starttime = clock();
	int query_all_neighbor_count = index.knnSearch(queryall, indices, dists, knn, flann::SearchParams(checks));
	printf("query_all_neighbor_count=%d\n", query_all_neighbor_count);
	printf("search knn done, %d ms\n", elapsedMilliseconds(starttime));
	printResult(dists, indices);
}

void main(int argc, char** argv)
{
	bool isSift = true;
	clock_t starttime = clock();
	string directory_prefix = "../../";
	string base_filename = directory_prefix + (isSift ? "sift_base.fvecs" : "gist_base.fvecs");
	string query_filename = directory_prefix + (isSift ? "sift_query.fvecs" : "gist_query.fvecs");
	string groundtruth_filename = directory_prefix + (isSift ? "sift_groundtruth.ivecs" : "gist_groundtruth.ivecs");
	Matrix<float> dataset = TexmexDataSetReader::readFMatrix(base_filename);
	Matrix<float> queryall = TexmexDataSetReader::readFMatrix(query_filename);
	printf("read data done, %d ms\n", elapsedMilliseconds(starttime));
	printf("data rows=%d, cols=%d\n", dataset.rows, dataset.cols);
	printf("query rows=%d, cols=%d\n", queryall.rows, queryall.cols);
	vector<vector<size_t>> groundTruthIndices = TexmexDataSetReader::readIvecs(groundtruth_filename);
	int knn = groundTruthIndices[0].size();
	printf("neighbor count=%d\n", knn);
	printf("ground truth:\n");
	printGroundTruthResult(queryall, dataset, groundTruthIndices);
	/*
	{
		Index<Distance> kdTreeIndex(dataset, KDTreeIndexParams(1));
		printf("kdtree:\n");
		buildAndSearch(kdTreeIndex, queryall, knn, 128);
	}
	*/
	{
		Index<Distance> lshIndex(dataset, LshIndexParams(2, 1000, 0));
		printf("lsh:\n");
		buildAndSearch(lshIndex, queryall, knn, -1);
	}
	delete[] dataset.ptr();
	delete[] queryall.ptr();
	system("pause");
}
