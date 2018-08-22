//
//  main.cpp
//  MultiThreaded matrix multiplication
//
//  compile command: g++ -std=c++11 ThreadMatMul.cpp  -o threadmatmul
//  run command: ./threadmatmul
//
//  Created by Taha Emara on 8/15/18.
//  Email: taha@emaraic.com
//

#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <mutex>


using namespace std;

#define synchronized(m) for(unique_lock<recursive_mutex> lk(m); lk; lk.unlock())

recursive_mutex m_mutex;


template <size_t rows>
void printMatrix(double (*mat)[rows], int r, int c){
    synchronized(m_mutex) {
        for (int i=0; i<r; ++i) {
            for (int j=0; j<c; ++j) {
                cout<<mat[i][j]<<" ";
            }
            cout<<"; "<<endl;
        }
    }
}

void printVector(vector<vector<double>> res){
    synchronized(m_mutex) {
        for (int ii=0; ii<res.size(); ++ii) {
            for (int j=0; j<res[0].size(); ++j) {
                cout<<res[ii][j]<<" ";
            }
            cout<<"; "<<endl;
        }
    }
}

template <size_t cols>
void initMatrix(double (*mat)[cols], int r, int c){
    for (int i=0; i<r; ++i) {
        for (int j=0; j<c; ++j) {
            mat[i][j]=((double) rand() / (RAND_MAX));
        }
    }
}

template <size_t cols, size_t cols1>
void * matrixMuliply(double (*a)[cols],double (*b)[cols1], const int m1r, const int m1c,const int m2r, const int m2c){
    double (*result)[cols1]=new double[m1r][cols1];
    for (int i=0; i<m1r; ++i) {
        for (int j=0; j<m2c; ++j) {
            double total=0;
            for (int k=0; k<m1c; ++k) {
                total+=a[i][k]*b[k][j];
            }
            result[i][j]=total;
        }
    }
    return result;
}

vector<vector<double>> vectorMuliply(vector<vector<double>> a,vector<vector<double>>b){
    if(a.size()!=b[0].size())
    {
        cerr<<"Number of cols of first mat not equal the number of rows of second mat"<<endl;
        exit(1);
    }
    vector<vector<double>>result(a.size(), vector<double>(b[0].size()));

//    synchronized(m_mutex) {
//        cout<<"========= Mat a ==========="<<endl;
//        printVector(a);
//        cout<<"========= Mat b ==========="<<endl;
//        printVector(b);
    
        for (int i=0; i<a.size(); ++i) {
            for (int j=0; j<b[0].size(); ++j) {
                double total=0;
                for (int k=0; k<a[0].size(); ++k) {
                    total+=a[i][k]*b[k][j];
                }
                result[i][j]=total;
            }
        }
//        cout<<"========= Result ==========="<<endl;
//        printVector(result);
//    }
    return result;
}

template <size_t cols, size_t cols1>
vector<vector<vector<double>>>  dividmatrix(double (*a)[cols],double (*b)[cols1], const int m1r, const int m1c,const int m2r, const int m2c, const int sections){
    vector<vector<vector<double>>> mats;
    
    int rows = m1r / sections;
    int rtotal = m1r % sections;
    int size = rows;
    for (int i = 0; i < sections; i++) {
        if (i == sections - 1) {
            size = size + rtotal;
        }
        vector<vector<double>> matrix(size, vector<double>(m1c));
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < m1c; k++) {
                matrix[j][k]=a[j+i*rows][k];
            }
        }
        mats.push_back(matrix);
    }
    
    
    int col = m2c/ sections;
    int ctotal = (m2c % sections);
    size = col;
    for (int i = 0; i < sections; i++) {
        if (i == sections - 1) {
            size = size + ctotal;
        }
        vector<vector<double>> matrix(m2r, vector<double>(size));
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < m2r; k++) {
                matrix[k][j]=b[k][j+i*col];
            }
        }
        mats.push_back(matrix);
    }
    
    return mats;
}



int main(int argc, const char * argv[]) {
    
    const int m1r=2000;//first matrix rows
    const int m1c=2000;//first matrix columns
    const int m2r=2000;//second matrix rows
    const int m2c=2000;//second matrix columns
    
    double (*a)[m1c]=new double[m1r][m1c];
    double (*b)[m2c]=new double[m2r][m2c];
    
    initMatrix(a, m1r, m1c);
    initMatrix(b, m2r, m2c);
    
    if(m1c!=m2r)
    {
        cerr<<"Number of cols of first mat not equal the number of rows of second mat"<<endl;
        exit(1);
    }
//    cout<<"------- Mat a ------ "<<endl;
//    printMatrix(a, m1r, m1c);
//    cout<<"------- Mat b ------ "<<endl;
//    printMatrix(b, m2r, m2c);
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    double (*res)[m2c]= (double (*)[m2c])matrixMuliply(a, b, m1r, m1c, m2r, m2c);
    chrono::steady_clock::time_point end= chrono::steady_clock::now();
    double s=chrono::duration_cast<chrono::milliseconds> (end - begin).count();
    cout<<"Ellapsed time for naive implementation :"<<s<<endl;
//    cout<<"------- Result ------ "<<endl;
//    printMatrix(res,m1r,m2c);
    cout<<"============================="<<endl;
    
    
    const int sections=8;
    std::thread threads[sections*sections];
    
    vector<vector<vector<double>>> mats= dividmatrix(a, b, m1r, m1c, m2r, m2c, sections);
    cout<<"result size "<<mats.size()<<endl;
    
    begin = chrono::steady_clock::now();
    
    for (int i=0; i<sections; ++i) {
        for (int j=sections; j<2*sections; ++j) {
            threads[(i*sections)+(j-sections)]=thread(vectorMuliply, mats[i], mats[j]);
        }
    }
    
    for (int z = 0; z < sections*sections; ++z) {
        threads[z].join();
    }
    
    end= chrono::steady_clock::now();
    s=chrono::duration_cast<chrono::milliseconds> (end - begin).count();
    cout<<"Ellapsed time for thread implementation :"<<s<<endl;
    
    
    return 0;
}
