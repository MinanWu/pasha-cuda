#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

class graph {

	public:
	int ALPHABET_SIZE;
	string ALPHABET;
	int* edgeVector;
    double* hittingNumVector;
    double* hittingNumAnyVector;
	int* topoSort;
	static map<char, int> alphabetMap;
    int k;
    int l;
    int edgeNum;
	int edgeCount;
    int vertexCount; 
    double** D;
    double** F;
    bool* used;
    bool* finished;
    double epsilon;
    double delta;
    int vertexExp;
    int vertexExp2;
    int vertexExp3;
    int vertexExpMask;
    int vertexExp_1;

	void generateGraph(int k) {      
        for (int i = 0; i < edgeNum; i++) edgeVector[i] = 1;
		edgeCount = edgeNum;
		vertexCount = edgeNum / ALPHABET_SIZE; 
    }

    char getChar(int i) {
    	return ALPHABET[i];
    }

    string getLabel(int i) {
		string finalString = "";
		for (int j = 0; j < k; j++) {
			finalString = getChar((i % ALPHABET_SIZE)) + finalString;
			i = i / ALPHABET_SIZE;
		}
		return finalString;
	}

	graph(int argK) {
		ALPHABET = "ACGT";
		ALPHABET_SIZE = 4;
		k = argK;
        edgeNum = pow(ALPHABET_SIZE, k);
        edgeVector = new int[edgeNum];
		generateGraph(k);
		map<char, int> alphabetMap;
		for (int i = 0; i < ALPHABET_SIZE; i++) alphabetMap.insert(pair<char,int>(ALPHABET[i], i));
    }

    void calculateForEach(int i, int L);
    void calculateForEachAny(int i);
    int calculateHittingNumber(int L);
    vector<int> calculateHittingNumberAny(int x);
    int calculateHittingNumberParallel(int L);
    vector<int> calculateHittingNumberParallelAny(int x);
    int calculatePaths(int L);
    int calculatePathsAny();
    vector<int> findMaxAny(int x);
    vector<int> getAdjacent(int v);
    int Hitting(int L, string hittingFile);
    int HittingAny(int L, int x, string hittingFile);
    int HittingParallel(int L, string hittingFile);
    int HittingParallelAny(int L, int x, string hittingFile);
    int HittingParallelRandom(int L, string hittingFile);
    int HittingParallelRandomAny(int L, int x, string hittingFile);
    int HittingRandom(int L, string hittingFile);
    int HittingRandomAny(int L, int x, string hittingFile);
    int maxLength();
    void removeEdge(int i);
    void topologicalSort();
    private:
	int depthFirstSearch(int index, int u);

};

#endif