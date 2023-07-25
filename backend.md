# Backend Optimization

This post gathers backend optimization techniques in machine learning.

## CPU

### GEMM

- [ ] [How to Optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [ ] [How to optimize GEMM on CPU](https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html)
- [ ] [GEMM: From Pure C to SSE Optimized Micro Kernels](https://www.mathematik.uni-ulm.de/~lehn/apfel/sghpc/gemm/)
- [ ] [如何利用TVM快速实现超越Numpy的GEMM](https://zhuanlan.zhihu.com/p/75203171)
- [ ] [x64 CPU GEMM 优化](https://zhuanlan.zhihu.com/p/593537184) ([玩转SIMD指令编程](https://zhuanlan.zhihu.com/p/591900754))
- [ ] [CPU高性能计算 1 - SGEMM 性能瓶颈分析与解决思路](https://zhuanlan.zhihu.com/p/604935952)
- [ ] [机器学习中的高性能计算（一）CPU优化](https://zhuanlan.zhihu.com/p/384654825)
- [ ] [机器学习中的高性能计算（二）SSE优化](https://zhuanlan.zhihu.com/p/409973153)

- [ ] [大佬是怎么优雅实现矩阵乘法的？](https://zhuanlan.zhihu.com/p/383115932)

## CUDA

### Elementwise operation

- [ ] [深入浅出GPU优化系列：elementwise优化及CUDA工具链介绍](https://zhuanlan.zhihu.com/p/488601925)
- [ ] [高效、易用、可拓展我全都要：OneFlow CUDA Elementwise 模板库的设计优化思路](https://zhuanlan.zhihu.com/p/447577193)
- [ ] [【BBuf 的CUDA笔记】一，解析OneFlow Element-Wise 算子实现](https://zhuanlan.zhihu.com/p/591058808)

### Reduction

- [ ] Chapter 10 of [*Programming Massively Parallel Processors*](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
- [ ] [如何实现一个高效的Softmax CUDA kernel？——OneFlow 性能优化分享](https://zhuanlan.zhihu.com/p/341059988)
- [ ] [【BBuf的CUDA笔记】八，对比学习OneFlow 和 FasterTransformer 的 Softmax Cuda实现](https://zhuanlan.zhihu.com/p/609198294)
- [ ] [CUDA高性能计算经典问题（一）—— 归约（Reduction）](https://zhuanlan.zhihu.com/p/416959273)
- [ ] [CUDA WarpReduce学习](https://zhuanlan.zhihu.com/p/492560229)
- [ ] [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)
- [ ] [简单谈谈CUDA Reduce](https://zhuanlan.zhihu.com/p/559549740)
- [ ] [CUDA编程入门（四）并行归约算法](https://zhuanlan.zhihu.com/p/98190609)
- [ ] [CUDA编程入门（五）更高效的并行归约算法](https://zhuanlan.zhihu.com/p/98416987)
- [ ] [CUDA编程入门（六）展开循环继续优化](https://zhuanlan.zhihu.com/p/98463812)
- [ ] [Pytorch CUDA源码解析 - BlockReduceSum](https://zhuanlan.zhihu.com/p/584936904)
- [ ] [【BBuf的CUDA笔记】八，对比学习OneFlow 和 FasterTransformer 的 Softmax Cuda实现](https://zhuanlan.zhihu.com/p/609198294)[^bing]

### Scan

- [ ] Chapter 11 of [*Programming Massively Parallel Processors*](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
- [ ] [高效CUDA Scan算法浅析](https://zhuanlan.zhihu.com/p/499963645)
- [ ] [CUB scan 算法学习](https://zhuanlan.zhihu.com/p/596332478)
- [ ] [CUDA高性能计算经典问题（二）—— 前缀和（Prefix Sum）](https://zhuanlan.zhihu.com/p/423992093)
- [ ] [Scan Primitives for GPU Computing](https://escholarship.org/uc/item/8051p6nd)

### GEMM/GEMV

- [ ] [传统 CUDA GEMM 不完全指北](https://zhuanlan.zhihu.com/p/584236348)
- [ ] [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)
- [ ] [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)[^gemm]
- [ ] [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175)
- [ ] [CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275)
- [ ] [深入浅出GPU优化系列：GEMM优化（一）](https://zhuanlan.zhihu.com/p/435908830)
- [ ] [深入浅出GPU优化系列：GEMM优化（二）](https://zhuanlan.zhihu.com/p/442930482)
- [ ] [深入浅出GPU优化系列：GEMM优化（三）](https://zhuanlan.zhihu.com/p/481600052)
- [ ] [如何开发机器学习系统：高性能GPU矩阵乘法](https://zhuanlan.zhihu.com/p/531498210)
- [ ] [CUDA Ampere Tensor Core HGEMM 矩阵乘法优化笔记 —— Up To 131 TFLOPS!](https://zhuanlan.zhihu.com/p/555339335)
- [ ] [有关CUBLAS中的矩阵乘法函数](https://www.cnblogs.com/cuancuancuanhao/p/7763256.html)
- [ ] [手把手推导分布式矩阵乘的最优并行策略](https://zhuanlan.zhihu.com/p/522759214)
- [ ] [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598)[^stream-k]

- [ ] [深入浅出GPU优化系列：gemv优化](https://zhuanlan.zhihu.com/p/494144694)
- [ ] [Sparse Matrix-Vector Multiplication with CUDA](https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f)
- [ ] [深入浅出GPU优化系列：spmv优化](https://zhuanlan.zhihu.com/p/509175679)
- [ ] [Accelerating Matrix Multiplication with Block Sparse Format and NVIDIA Tensor Cores](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)
- [ ] [Sparse GPU Kernels for Deep Learning](https://arxiv.org/abs/2006.10901)
- [ ] [GPU Kernels for Block-Sparse Weights](https://openai.com/research/block-sparse-gpu-kernels)
- [ ] [Block Sparse Matrix-Vector Multiplication with CUDA](https://medium.com/gpgpu/block-sparse-matrix-vector-multiplication-with-cuda-4e616b30267)[^correction]

### Convolution

- [ ] Chapters 7 and 16 of [*Programming Massively Parallel Processors*](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
- [ ] [MegEngine TensorCore 卷积算子实现原理](https://zhuanlan.zhihu.com/p/372973726)
- [ ] [卷积神经网络性能优化](https://zhuanlan.zhihu.com/p/80361782)
- [ ] [Im2Col+GEMM的改进方法MEC，一种更加高效的卷积计算策略](https://zhuanlan.zhihu.com/p/264554159)
- [ ] [MegEngine Inference 卷积优化之 Im2col 和 winograd 优化](https://zhuanlan.zhihu.com/p/532187602)
- [ ] [CUDA卷积算子手写详细实现](https://zhuanlan.zhihu.com/p/613538649)

### Layer

- [ ] [CUDA优化之LayerNorm性能优化实践](https://zhuanlan.zhihu.com/p/443026261)
- [ ] [【BBuf的CUDA笔记】二，解析OneFlow BatchNorm相关算子实现](https://zhuanlan.zhihu.com/p/593483751)
- [ ] [【BBuf的CUDA笔记】六，总结 FasterTransformer Encoder(BERT) 的cuda相关优化技巧](https://zhuanlan.zhihu.com/p/601130731)
- [ ] [【BBuf的CUDA笔记】七，总结 FasterTransformer Decoder(GPT) 的cuda相关优化技巧](https://zhuanlan.zhihu.com/p/603611192)

### Miscellaneous

- [ ] [【BBuf的CUDA笔记】四，介绍三个高效实用的CUDA算法实现（OneFlow ElementWise模板，FastAtomicAdd模板，OneFlow UpsampleNearest2d模板）](https://zhuanlan.zhihu.com/p/597435971)
- [ ] [如何实现比PyTorch快6倍的Permute/Transpose算子？](https://zhuanlan.zhihu.com/p/425587014)[^permute]
- [ ] [在OneFlow实现Unfold Fold算子](https://zhuanlan.zhihu.com/p/418191393)[^unfold]
- [ ] [实例：手写 CUDA 算子，让 Pytorch 提速 20 倍（某特殊算子）](https://zhuanlan.zhihu.com/p/476297195)
- [ ] [CUDA GroupNorm NHWC优化](https://zhuanlan.zhihu.com/p/596871310)

## Framework

- [ ] [The Journey of an Operator in a Deep Learning Framework](https://medium.com/codex/the-journey-of-an-operator-in-a-deep-learning-framework-60d404750cb1)
- [ ] [OneFlow源码解析：自动微分机制](https://weibo.com/ttarticle/p/show?id=2309404841463233249338)

## Profiling

- [ ] [深入浅出GPU优化系列：elementwise优化及CUDA工具链介绍](https://zhuanlan.zhihu.com/p/488601925)

## Customized PyTorch kernel

- [ ] [Official tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [ ] [像教女朋友一样教你用Cuda实现PyTorch算子](https://zhuanlan.zhihu.com/p/595851188)
- [ ] [PyTorch自定义CUDA算子教程与运行时间分析](https://zhuanlan.zhihu.com/p/358220419)
- [ ] [详解PyTorch编译并调用自定义CUDA算子的三种方式](https://zhuanlan.zhihu.com/p/358778742)
- [ ] [三分钟教你如何PyTorch自定义反向传播](https://zhuanlan.zhihu.com/p/359524837)
- [ ] [PyTorch 源码解读之 cpp_extension：揭秘 C++/CUDA 算子实现和调用全流程](https://zhuanlan.zhihu.com/p/348555597)

[^bing]: [【BBuf的CUDA笔记】九，使用newbing（chatgpt）解析oneflow softmax相关的fuse优化](https://zhuanlan.zhihu.com/p/615619524)

[^gemm]: The code is available at https://github.com/niuhope/cuda_sgemm.

[^stream-k]: The [code](https://github.com/NVIDIA/cutlass/tree/main/examples/47_ampere_gemm_universal_streamk) is now part of [cuTLASS](https://github.com/NVIDIA/cutlass).

[^correction]: [使用CUDA实现块稀疏矩阵向量乘（BSpMV）](https://zhuanlan.zhihu.com/p/620575933)

[^permute]: The code is available at https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/ep/cuda/primitive/permute.cu.

[^unfold]: PyTorch [`nn.Unfold`](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html) generalizes the $\verb|im2col|$ operation.
