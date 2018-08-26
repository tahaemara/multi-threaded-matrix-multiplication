package com.emaraic.cuda;

import java.util.Arrays;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import static org.bytedeco.javacpp.cublas.*;

/**
 * Matrix multiplication using Cuda, before running this class, you must have a
 * cuda enabled GPU and install cuda 9.0.
 *
 * Created on Aug 24, 2018 , 6:48:55 PM
 *
 * @author Taha Emara 
 * Email : taha@emaraic.com 
 * Website: http://www.emaraic.com
 */
class MatrixMultiplicationJavaCppCuda {

    /**
     * initialize the input matrix randomly
     *
     * @param mat input matrix
     */
    public static void initMat(float mat[]) {
        for (int i = 0; i < mat.length; i++) {
            mat[i] = (float) Math.random();
        }
    }

    /**
     * print the input matrix
     *
     * @param a input matrix
     */
    private static void printMat(float[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + " ");
        }
        System.out.println("");
    }

    /**
     * Creates a String from a zero-terminated string in a byte array
     *
     * @param bytes The byte array
     * @return The String
     */
    private static String createString(byte bytes[]) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bytes.length; i++) {
            char c = (char) bytes[i];
            if (c == 0) {
                break;
            }
            sb.append(c);
        }
        return sb.toString();
    }

    public static void main(String args[]) {

        System.out.println(Float.BYTES);
        int m1r = 2000;//first matrix rows
        int m1c = 2000;//first matrix columns
        int m2r = 2000;//second matrix rows
        int m2c = 2000;//second matrix columns

        float alpha = 1.0f;
        float beta = 0.0f;

        FloatPointer pa = new FloatPointer();
        FloatPointer pb = new FloatPointer();
        FloatPointer pc = new FloatPointer();

        float a[] = new float[m1r * m1c]; //m*k
        float b[] = new float[m2r * m2c]; //k*n
        initMat(a);
        initMat(b);

        /*printMat(a);
        printMat(b);*/
        float c[] = new float[m1r * m2c];//m*n

        /* Allocate device memory for the matrices */
        cublasAlloc(a.length, Float.BYTES, pa);
        cublasAlloc(b.length, Float.BYTES, pb);
        cublasAlloc(c.length, Float.BYTES, pc);

        /* Initialize the device matrices with the host matrices */
        cublasSetVector(a.length, Float.BYTES, new FloatPointer(a), 1, pa, 1);
        cublasSetVector(b.length, Float.BYTES, new FloatPointer(b), 1, pb, 1);
        cublasSetVector(c.length, Float.BYTES, new FloatPointer(c), 1, pc, 1);


        int m = m1r; // A.numRows();
        int n = m2c; // B.numColumns();
        int k = m1c; // A.numColumns();

        int lda = m1r; // leading diemsion of matrix a, which is number of rows;
        int ldb = m2r;
        int ldc = m1r;

        long start = System.nanoTime();
        //real sinle-precision (float) multiplication, for double-precision (double) use JCublas.cublasDgemm
        char nn = 'N';
        cublasSgemm((byte) nn, (byte) nn, m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
        System.out.println("Cublas implementation:" + (System.nanoTime() - start) / 1000000.0);

        /* Read the result back */
        FloatPointer ctemp = new FloatPointer(c);
        cublasGetVector(c.length, Float.BYTES, pc, 1, ctemp, 1);

        /*Print result*/
        /* for (int i = 0; i < c.length; i++) {
        float d = ctemp.get(i);
        System.out.print(d + " ");
        }*/

        /* Memory clean up */
        cublasFree(pa);
        cublasFree(pb);
        cublasFree(pc);

        /* Shutdown */
        cublasShutdown();

    }

}
