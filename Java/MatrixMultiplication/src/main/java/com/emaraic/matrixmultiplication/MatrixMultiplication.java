package com.emaraic.matrixmultiplication;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: Aug 9, 2018
 */
public class MatrixMultiplication {

    /**
     * initialize the input matrix randomly
     *
     * @param mat input matrix
     */
    public static void initMat(double mat[][]) {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                mat[i][j] = Math.random();
            }
        }
    }

    /**
     * multiply two input matrices
     *
     * @param a first input matrix
     * @param b second input matrix
     * @return the result of the multiplication of matrix a and matrix b
     */
    public static double[][] matrixMul(double a[][], double b[][]) {
        if (a[0].length != b.length) {
            throw new ArithmeticException("Number of cols of first mat not equal the number of rows of second mat");
        }
        double result[][] = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                double total = 0;
                for (int k = 0; k < b.length; k++) {
                    total += a[i][k] * b[k][j];
                }
                result[i][j] = total;
            }
        }
        return result;
    }

    /**
     * Print the input matrix to stdout
     *
     * @param mat input matrix
     */
    public static void printMat(double mat[][]) {
        for (double[] mat1 : mat) {
            for (int j = 0; j < mat1.length; j++) {
                System.out.printf("%.2f ", mat1[j]);
            }
            System.out.println("");
        }
        System.out.println("");
    }

    /**
     * divide first matrix into sections horizontally and second matrix vertically.
     * 
     * @param a first input matrix
     * @param b second input matrix
     * @param sections number of sub-matrices for every input matrix
     * @return lists of  mat "a" sections and mat "b" sections
     */
    public static List<List<double[][]>> divideMat(double[][] a, double[][] b, int sections) {
        List<double[][]> amats = new ArrayList<>();
        int rows = a.length / sections;
        int rtotal = a.length % sections;
        int size = rows;
        for (int i = 0; i < sections; i++) {
            if (i == sections - 1) {
                size = size + rtotal;
            }
            double temp[][] = new double[size][a[0].length];
            
            System.arraycopy(a, rows * i, temp, 0, size);
            /*printMat(temp);
            System.out.println("++++++++++++");*/
            amats.add(temp);
        }

        List<double[][]> bmats = new ArrayList<>();
        int cols = b[0].length / sections;
        int ctotal = (b[0].length % sections);
        size = cols;
        for (int i = 0; i < sections; i++) {
            if (i == sections - 1) {
                size = size + ctotal;
            }
            double temp[][] = new double[b.length][size];
            for (int j = 0; j < b.length; j++) {
                System.arraycopy(b[j], cols * i, temp[j], 0, size);
            }
            /*printMat(temp);
            System.out.println("++++++++++++");*/
            bmats.add(temp);
        }
        List<List<double[][]>> combined = new ArrayList<>();
        combined.add(amats);
        combined.add(bmats);
        return combined;
    }


    public static void main(String[] args) {

        double a[][] = new double[2000][2000];
        double b[][] = new double[2000][2000];
        initMat(a);
        initMat(b);
        
        //printMat(a);
        //printMat(b);

        /*Test naive implementation*/
        long start = System.currentTimeMillis();
        /*        double[][] matrixMul = matrixMul(a, b);
        //printMat(matrixMul);
        System.out.println("Naive implementation:" + (System.currentTimeMillis() - start));
        */

        /*Tesing thread implementation*/
        int sections = 4;
        MatrixThread[] threads = new MatrixThread[sections*sections];
        List<List<double[][]>> div = divideMat(a, b, sections);
        List<double[][]> amats = div.get(0);
        List<double[][]> bmats = div.get(1);
        start = System.currentTimeMillis();
        for (int i = 0; i < amats.size(); i++) {
            double[][] get = amats.get(i);
            for (int j = 0; j < bmats.size(); j++) {
                double[][] get1 = bmats.get(j);
                MatrixThread m = new MatrixThread(get, get1);
                threads[i*sections+j] = m;
                m.start();
                //double[][] matrixMul1 = matrixMul(get, get1);
                //printMat(mt[i].getResult());
            }
        }
        
        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException ex) {
                System.out.println(ex.getMessage());
            }
        }
        
        /*for (MatrixThread t : threads) {
        double[][] result = t.getResult();
        printMat(result);
        System.out.println("");
        }*/
        
        System.out.println("Multi-threaded implementation:" + (System.currentTimeMillis() - start));
        System.out.println("---------------------");

        
        /*Test ND4l implementation*/
        INDArray create = Nd4j.create(a);
        INDArray create1 = Nd4j.create(b);
        start = System.currentTimeMillis();
        INDArray mmul = create.mmul(create1);
        System.out.println("Nd4j implementation: " + (System.currentTimeMillis() - start));
        //System.out.println(mmul);

        start = System.currentTimeMillis();
        INDArray mmul2 = Nd4j.gemm(create, create1, false, false);
        System.out.println("Nd4j geem implementation: " + (System.currentTimeMillis() - start));
        //System.out.println(mmul2);
        System.out.println(mmul.equalsWithEps(mmul2, 1));
    }

}
