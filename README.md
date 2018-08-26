# Multi-Threaded Matrix Multiplication

Multiplication of matrices is a core operation in neural network and especially deep learning. So, this repository contains an implementation of the naive and multi-threaded matrix multiplication in Java and C++ and also comparing then with the Nd4j library(Openblas backend) and also Cuda with GPU Nvidia GTX 1080 ti.

<table class="table table-bordered table-striped" style="margin: 0 auto !important;float: none !important;width: auto;"> 
  <caption>This test was done with matrix 2000x2000 on 2.2 GHz Intel Core i7</caption>
<thead> 									
	<tr><td></td> <td>Java</td> <td>C++</td> <td>Nd4j</td> <td>Jcuda</td></tr> </thead>
	 <tbody> 
     	 <tr> <td>Naive</td> <td>154615 ms</td> <td>90288 m </td> <td rowspan="2">129 ms </td><td rowspan="2">0.093032 ms</td></tr> 
	 <tr> <td>Multi-Threaded</td> <td>21307 ms</td> <td>51504 ms </td>  </tr> 
</tbody></table>




### Recommended Reading

https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

