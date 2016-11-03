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

vector<int> computePrecision(Matrix<size_t> indice, vector<vector<size_t>> ground) {
	vector<int> count;
	for (size_t row = 0; row < indice.rows; row++) {
		int count_com = 0;
		for (size_t col = 0; col < indice.cols; col++) {
			for (size_t identi = 0; identi < ground[0].size(); identi++) {
				if (indice[row][col] == ground[row][identi]) {
					count_com++;
					break;
				}
			}
		}
		count.push_back(count_com);
	}
	return count;
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
		//printf("%f\n", sum);
		averageMAP = sum / indice.cols;
		//printf("%f\n", averageMAP);
		queryMAP.push_back(averageMAP);
	}
	for (int i = 0; i < queryMAP.size(); i++) {
		//printf("%f\n", queryMAP[i]);
		sumMAP += queryMAP[i];
	}
	printf("%f\n", sumMAP);
	printf("%d\n", queryMAP.size());
	printf("%f\n", sumMAP / queryMAP.size());
	return sumMAP / queryMAP.size();
}

//void printStatistics(const vector<vector<float>>& distancesVector, int knn) {
//	vector<float> minDisVector;
//	vector<float> maxDisVector;
//	vector<float> meanDisVector;
//	vector<float> stdVarDistanceVector;
//	for (size_t neighbor_index = 0; neighbor_index < knn; ++neighbor_index) {
//		minDisVector.push_back(FLT_MAX/10);
//		maxDisVector.push_back(-FLT_MAX/10);
//		meanDisVector.push_back(0);
//		stdVarDistanceVector.push_back(0);
//	}
//	for (size_t neighbor_index = 0; neighbor_index < knn; ++neighbor_index) {
//		for (size_t query_row = 0; query_row < distancesVector.size(); ++query_row) {
//			if (neighbor_index < distancesVector[query_row].size()) {
//				float distance = distancesVector[query_row][neighbor_index];
//				minDisVector[neighbor_index] = min(minDisVector[neighbor_index], distance);
//				maxDisVector[neighbor_index] = max(maxDisVector[neighbor_index], distance);
//				meanDisVector[neighbor_index] += distance;
//			}
//		}
//		meanDisVector[neighbor_index] /= distancesVector.size();
//	}
//	for (size_t neighbor_index = 0; neighbor_index < knn; ++neighbor_index) {
//		for (size_t query_row = 0; query_row < distancesVector.size(); ++query_row) {
//			if (neighbor_index < distancesVector[query_row].size()) {
//				float distance = distancesVector[query_row][neighbor_index];
//				float diff = distance - meanDisVector[neighbor_index];
//				stdVarDistanceVector[neighbor_index] += diff*diff;
//			}
//		}
//		stdVarDistanceVector[neighbor_index] /= distancesVector.size() - 1 == 0 ? 1 : distancesVector.size() - 1;
//		stdVarDistanceVector[neighbor_index] = sqrt(stdVarDistanceVector[neighbor_index]);
//	}
//
//	for (size_t neighbor_index = 0; neighbor_index < print_neighbor_count; ++neighbor_index) {
//		printf("neighbor_index=%d, min=%f, max=%f, mean=%f, stdVar=%f\n", neighbor_index, minDisVector[neighbor_index], maxDisVector[neighbor_index], meanDisVector[neighbor_index], stdVarDistanceVector[neighbor_index]);
//	}
//}

//void printGroundTruthResult(const Matrix<float>& query, const Matrix<float>& baseMatrix, const vector<vector<size_t>>& groundTruthIndices) {
//	vector<vector<float>> groundTruthDistances;
//	Distance distance_functor;
//	for (int query_row = 0; query_row < query.rows; query_row++) {
//		vector<float> distances;
//		float* query_feature = new float[query.cols];
//		for (size_t i = 0; i < query.cols; ++i) {
//			query_feature[i] = query[query_row][i];
//		}
//		for (int neighbor_index = 0; neighbor_index < groundTruthIndices[query_row].size(); neighbor_index++) {
//			float* groundtruth_feature = new float[query.cols];
//			for (size_t j = 0; j < query.cols; ++j) {
//				groundtruth_feature[j] = baseMatrix[groundTruthIndices[query_row][neighbor_index]][j];
//			}
//			float distance = sqrt(distance_functor(query_feature, groundtruth_feature, query.cols));
//			distances.push_back(distance);
//			delete[] groundtruth_feature;
//		}
//		sort(distances.begin(), distances.end());
//		delete[] query_feature;
//		groundTruthDistances.push_back(distances);
//	}
//	printStatistics(groundTruthDistances, baseMatrix.cols);
//}

//void printResult(const flann::Matrix<float>& dists, const Matrix<size_t>& indices) {
//	vector<vector<float>> floatDistances;
//	for (int i = 0; i < dists.rows; i++) {
//		floatDistances.push_back(vector<float>());
//		for (int j = 0; j < dists.cols; j++) {
//			if (indices[i][j] != -1) {
//				floatDistances[i].push_back(sqrt(dists[i][j]));
//			}
//		}
//		sort(floatDistances[i].begin(), floatDistances[i].end());
//	}
//	printStatistics(floatDistances, indices.cols);
//}

void buildAndSearch(Index<Distance> index, const Matrix<float>& queryall, int knn, int checks, vector<vector<size_t>> groundTruthIndice) {
	int minCount = FLT_MAX;
	int maxCount = -FLT_MAX;
	int meanCount;
	int sumCount = 0;
	int sum = 0;
	int varianceCount;
	Matrix<size_t> indices(new size_t[queryall.rows*knn], queryall.rows, knn);
	Matrix<float> dists(new float[queryall.rows*knn], queryall.rows, knn);
	memset(indices.ptr(), (size_t)-1, queryall.rows*knn*sizeof(size_t));
	clock_t starttime = clock();
	index.buildIndex();
	printf("build index done, %d ms\n", elapsedMilliseconds(starttime));
	starttime = clock();
	int query_all_neighbor_count = index.knnSearch(queryall, indices, dists, knn, flann::SearchParams(checks));
	printf("search knn done, %d ms\n", elapsedMilliseconds(starttime));
	vector<int> com = computePrecision(indices, groundTruthIndice);
	for (int i = 0; i < com.size(); i++) {
		//printf("the %d query of groungtruth is %d.\n", i, com[i]);
		if (com[i] < minCount) {
			minCount = com[i];
		}
		if (com[i] > maxCount) {
			maxCount = com[i];
		}
		sumCount += com[i];
	}
	meanCount = sumCount / com.size();
	for (int i = 0; i < com.size(); i++) {
		sum += (com[i] - meanCount)*(com[i] - meanCount);
    }
	varianceCount = sum / com.size();
	printf("min count: %d\n", minCount);
	printf("max count: %d\n", maxCount);
	printf("sum count: %d\n", sumCount);
	printf("mean count: %d\n", meanCount);
	printf("MAP: %f\n", computeMAP(indices, groundTruthIndice));
	//printf("variance count: %d\n", varianceCount);
	//printResult(dists, indices);
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
	//printf("ground truth:\n");
	//printGroundTruthResult(queryall, dataset, groundTruthIndices);
	{
		Index<Distance> kdTreeIndex(dataset, KDTreeIndexParams(1));
		printf("kdtree:\n");
		buildAndSearch(kdTreeIndex, queryall, knn, 128, groundTruthIndices);
	}
	
	{
		Index<Distance> lshIndex(dataset, LshIndexParams(30, 2000, 0));
		printf("lsh:\n");
		buildAndSearch(lshIndex, queryall, knn, -1, groundTruthIndices);
	}
	delete[] dataset.ptr();
	delete[] queryall.ptr();
	system("pause");
}
