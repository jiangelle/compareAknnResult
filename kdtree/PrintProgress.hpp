#pragma once
#include <stdio.h>

class PrintProgress {
public:
	static void printProgress(double progress) {
		printf("[");
		const int total = 100;
		int progressInt = progress * total;
		for (int i = 0; i < progressInt; ++i) {
			printf("=");
		}
		printf("%s", progressInt == total ? "=" : ">");
		for (int i = 0; i < total - progressInt; ++i) {
			printf(" ");
		}
		printf("]\r");
	}
};