//
//  NaiveMatMul.cpp
//  Naive Matrix Multiplication
//
//  compile command: g++ -std=c++11 NaiveMatMul.cpp  -o matmul
//  run command: ./matmul
//
//  Created by Taha Emara on 8/16/18.
//  Email: taha@emaraic.com
//

#include <iostream>
#include <chrono>

using namespace std;

template <size_t rows>
void printMatrix(double (*mat)[rows], int r, int c){
    for (int i=0; i<r; ++i) {
        for (int j=0; j<c; ++j) {
            cout<<mat[i][j]<<" ";
        }
        cout<<endl;
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

int main(int argc, const char * argv[]) {
    
    const int m1r=2000;//first matrix rows
    const int m1c=2000;//first matrix columns
    const int m2r=2000;//second matrix rows
    const int m2c=2000;//second matrix columns
    
    double (*a)[m1c]=new double[m1r][m1c];
    double (*b)[m2c]=new double[m2r][m2c];
    
    initMatrix(a, m1r, m1c);//initiate matrix with random numbers between 0 & 1
    initMatrix(b, m2r, m2c);
    
    
    //for small sizes, you can uncomment the following lines to print the matrices
    //printMatrix(a, m1r, m1c);
    //cout<<"-------------"<<endl;
    //printMatrix(b, m2r, m2c);
    //cout<<"-------------"<<endl;
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    double (*result)[m2c]= (double (*)[m2c])matrixMuliply(a, b, m1r, m1c, m2r, m2c);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    double s=std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
    
    cout<<"It takes: "<<s<<"ms"<<endl;
    
    //printMatrix(result,m1r,m2c);
    
    delete [] result;
    delete [] a;
    delete []b;
    return 0;
}
