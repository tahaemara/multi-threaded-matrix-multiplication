package com.emaraic.matrixmultiplication;

/**
 *
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: Aug 10, 2018
 */
public class MatrixThread extends Thread {

    private double a[][];
    private double b[][];
    private double result[][];

    public MatrixThread(double a[][], double b[][]) {
        this.a = a;
        this.b = b;
    }

    /**
     * multiply two input matrices
     *
     * @param a first input matrix
     * @param b second input matrix
     * @return the result of the multiplication of matrix a and matrix b
     */
    private double[][] matrixMul(double a[][], double b[][]) {
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

    @Override
    public void run() {
        
            result = matrixMul(a, b);
        try {
            Thread.sleep(10);
        } catch (InterruptedException ex) {
            System.out.println(ex.getMessage());    
        }
    }

    /**
     *
     * @return result matrix
     */
    public double[][] getResult() {
        return result;
    }

}
