#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <flann\util\matrix.h>
using namespace std;
using namespace flann;

class TexmexDataSetReader {
public:
	static Matrix<float> readFMatrix(const string filename) {
		vector<vector<float>> vecvec;
		ifstream in(filename, ios::binary);
		if (!in.is_open()) {
			throw filename + " not open";
		}
		while (!in.eof()) {
			int dimension;
			in.read((char*)&dimension, sizeof(dimension));
			vector<float> vec;
			for (size_t i = 0; i < dimension; ++i) {
				float value;
				in.read((char*)&value, sizeof(value));
				vec.push_back(value);
				//printf("%f%s", value, i == dimension-1 ? "\n" : ",");
			}
			vecvec.push_back(vec);
		}
		vecvec.pop_back();
		///////////////////
		int rows = vecvec.size();
		int cols = vecvec[0].size();
		Matrix<float> matrix(new float[rows*cols], rows, cols);
		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				matrix[row][col] = vecvec[row][col];
			}
		}
		return matrix;
	}
	static vector<vector<size_t>> readIvecs(const string filename) {
		vector<vector<size_t>> vecvec;
		ifstream in(filename, ios::binary);
		if (!in.is_open()) {
			throw filename + " not open";
		}
		while (!in.eof()) {
			int dimension;
			in.read((char*)&dimension, sizeof(dimension));
			vector<size_t> vec;
			for (size_t i = 0; i < dimension; ++i) {
				int value;
				in.read((char*)&value, sizeof(value));
				vec.push_back(value);
				//printf("%d%s", value, i == dimension-1 ? "\n" : ",");
			}
			vecvec.push_back(vec);
		}
		vecvec.pop_back();
		return vecvec;
	}
};
