
This is an experiment that tried to analyse the inception behind inception net. That is, factorising the convolutions. 
Can a sequence of  nx1 -> 1xn convolution be more efficient than a single nxn convolution...
The answer lies in this article

To benchmark this I have run a number of models in both cpu and gpu of my PC and you may find the results here...

What I found was, the effective efficiency is only theoretical and not practical for a variety of reasons.

From my experiments, it is evident that a single nxn convolution is twice as fast as a sequence of nx1->1xn convolutions. Ofcourse, having a nonlinearity introduced inbetween shows to be slightly better even though there are less number of parameters. 

The reasons for this to not work as expected are:

1. Sequential Operations: Doubling the Computational Load
3x3 Convolution: A single 3x3 convolution involves sliding a 3x3 kernel over the entire input feature map. This operation is completed in one pass.
1x3 and 3x1 Convolutions: When you split the 3x3 convolution into a 1x3 followed by a 3x1 convolution, you are effectively performing two separate convolutional operations.
First Operation (1x3): The 1x3 kernel slides across each row of the input feature map.
Second Operation (3x1): The 3x1 kernel then slides across each column of the resulting feature map from the first operation.
These operations are performed sequentially, meaning that after completing the 1x3 convolution, you then move on to the 3x1 convolution. As a result, the data is processed twice, doubling the amount of work required compared to a single 3x3 convolution, which processes the data in one go.
You may think that this is not gonna affect the speed because there are lesser number of parameters to be learnt in a sequence of  n*1->1*n. If you think so, the second reason will obviously change your mind.


2. Memory Access Patterns and Efficiency
3x3 Convolution: During a 3x3 convolution, the kernel accesses a contiguous block of memory, which is efficient for the CPU/GPU to handle. Modern hardware and software libraries are optimized to perform these operations quickly.
1x3 and 3x1 Convolutions: In the split approach, the 1x3 convolution accesses memory across rows, and the 3x1 convolution accesses memory across columns. This can lead to less efficient memory access patterns because the data might not be stored contiguously in memory after the first operation. The second operation might require fetching data from different parts of memory, leading to increased memory access times and potential cache misses, further slowing down the computation.
Well... this is how the OS works and this is how python is gonna work.


3. Increased Number of Operations
Operation Count:
In a single 3x3 convolution, the operation count is determined by the number of elements in the kernel (3x3=9) multiplied by the number of input channels and the number of output filters.
In the case of 1x3 and 3x1 convolutions, you first apply the 1x3 kernel (3 operations per input pixel) and then apply the 3x1 kernel (another 3 operations per input pixel). While each operation is simpler, you are effectively performing 6 operations per input pixel (3 for 1x3 and 3 for 3x1), instead of 9 in a single 3x3 convolution. However, due to the sequential nature, you end up with a larger overall operation count across the entire input feature map, leading to increased computational demands.
This can be related with the second reason.


4.  Caching and Memory Bandwidth
3x3 Convolution: The data needed for a 3x3 convolution is often small enough to fit in the GPU cache, meaning that the operation can be performed quickly with minimal data fetching from slower main memory.
1x3 and 3x1 Convolutions: These operations might not fit as well into cache, especially because the output of the 1x3 convolution needs to be stored and then reloaded for the 3x1 convolution. This increases the demand on memory bandwidth and can lead to additional delays as data is fetched from main memory.


Apart from these four reasons, I also think that there may be two more reasons for the experiment's results.
1. GPU/Hardware Optimization
Optimization for 3x3 Convolutions: Most deep learning frameworks and hardware accelerators (like GPUs) are highly optimized for standard convolutional operations, especially common kernel sizes like 3x3. These optimizations include parallel processing, specialized instruction sets, and efficient memory usage.
Lack of Optimization for Split Convolutions: The split 1x3 and 3x1 convolutions might not benefit from these optimizations to the same extent. The hardware might not be able to parallelize these operations as effectively as it does for a 3x3 convolution. This leads to less efficient use of computational resources, which in turn increases training time.
I would want a cuda expert to comment on this ofcourse. But to check if this reason is true, I tried a 6*7 kernel too and ended with the same results. i.e, factorized convolutions being twice slow. Anyhow, the same results for CPU too.


2. Pipeline Stalls and Synchronization Overheads
Pipeline Processing: In deep learning, operations are often pipelined to maximize hardware utilization. For example, while one layer is being processed by the GPU, data for the next layer is being fetched or pre-processed.
Impact of Sequential Layers: When you use sequential layers like 1x3 followed by 3x1, the second operation has to wait for the first to complete. This can cause pipeline stalls, where the GPU is underutilized because it has to wait for data from the previous operation. This is less of an issue with a single 3x3 convolution, which can be pipelined more effectively.
This comes here only because I havent used any data generator/loader objects here to batch and load data from ROM to RAM. Also, I havent used prefetch for the GPU side experiments.


So, on the whole, I do not exactly know how people train those kind of very large networks on large servers. But what I do know is that, they would be using a number of GPUs and higly optimised advanced distributed training methodologies. 
So, this factorising technique may work (ofcourse it has) for them, but for enthusiasts like us using PCs, it is better to use smaller number of layers for the above mentioned reasons.
