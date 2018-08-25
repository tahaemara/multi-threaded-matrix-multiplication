package com.emaraic.jcuda;

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
import jcuda.*;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;
import static jcuda.driver.JCudaDriver.cuInit;
import jcuda.jcublas.*;

class MatrixMultiplicationCuda {

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
        
        //Print device name
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        byte deviceName[] = new byte[1024];
        cuDeviceGetName(
                deviceName, deviceName.length, device);
        String name = createString(deviceName);
        System.out.println("Device Info: \n" + name);
        
        int m1r = 2000;//first matrix rows
        int m1c = 2000;//first matrix columns
        int m2r = 2000;//second matrix rows
        int m2c = 2000;//second matrix columns

        Pointer pa = new Pointer();
        Pointer pb = new Pointer();
        Pointer pc = new Pointer();

        float alpha = 1.0f;
        float beta = 0.0f;

        /* Initialize JCublas */
        JCublas.cublasInit();

        float a[] = new float[m1r * m1c]; //m*k
        float b[] = new float[m2r * m2c]; //k*n
        initMat(a);
        initMat(b);
        /*printMat(a);
        printMat(b);*/

        float c[] = new float[m1r * m2c];//m*n


        /* Allocate device memory for the matrices */
        JCublas.cublasAlloc(a.length, Sizeof.FLOAT, pa);
        JCublas.cublasAlloc(b.length, Sizeof.FLOAT, pb);
        JCublas.cublasAlloc(c.length * 2000, Sizeof.FLOAT, pc);

        /* Initialize the device matrices with the host matrices */
        JCublas.cublasSetVector(a.length, Sizeof.FLOAT, Pointer.to(a), 1, pa, 1);
        JCublas.cublasSetVector(b.length, Sizeof.FLOAT, Pointer.to(b), 1, pb, 1);
        JCublas.cublasSetVector(c.length, Sizeof.FLOAT, Pointer.to(c), 1, pc, 1);

        int m = m1r; // A.numRows();
        int n = m2c; // B.numColumns();
        int k = m1c; // A.numColumns();

        int lda = m1r; // leading diemsion of matrix a, which is number of rows;
        int ldb = m2r;
        int ldc = m1r;

        long start = System.nanoTime();
        //real sinle-precision (float) multiplication, for double-precision (double) use JCublas.cublasDgemm
        JCublas.cublasSgemm('n', 'n', m, n, k, alpha,
                pa, lda, pb, ldb, beta, pc, ldc);
        System.out.println("Cublas implementation:" + (System.nanoTime() - start) / 1000000.0);

        /* Read the result back */
        JCublas.cublasGetVector(c.length, Sizeof.FLOAT, pc, 1, Pointer.to(c), 1);

        /* Memory clean up */
        JCublas.cublasFree(pa);
        JCublas.cublasFree(pb);
        JCublas.cublasFree(pc);

        /* Shutdown */
        JCublas.cublasShutdown();

        //System.out.println(Arrays.toString(c));
    }

}
