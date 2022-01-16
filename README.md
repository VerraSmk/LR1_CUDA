# LR1_CUDA
Был написан параллельный алгоритм перемножения матриц с использованием разбиение матрицы на матрицы 16*16 и работу уже с ними + использовалась shared память. 

Алгоритм для CPU стандартный.
Язык программирования: С/С++
IDE: VS2019

В ходе	эксперемента использовалось
Experiment for matrix size: 64
Time spent executing by the CPU: 0.00 millseconds
Time spent executing by the GPU events: 0.22 millseconds
Time spent executing by the GPU: 1.00 millseconds
Acceleration factor: 0.00
Relevance: true
Experiment for matrix size: 128
Time spent executing by the CPU: 6.00 millseconds
Time spent executing by the GPU events: 0.39 millseconds
Time spent executing by the GPU: 1.00 millseconds
Acceleration factor: 6.00
Relevance: true
Experiment for matrix size: 256
Time spent executing by the CPU: 49.00 millseconds
Time spent executing by the GPU events: 2.07 millseconds
Time spent executing by the GPU: 3.00 millseconds
Acceleration factor: 16.33
Relevance: true
Experiment for matrix size: 512
Time spent executing by the CPU: 472.00 millseconds
Time spent executing by the GPU events: 13.75 millseconds
Time spent executing by the GPU: 15.00 millseconds
Acceleration factor: 31.47
Relevance: true
Experiment for matrix size: 1024
Time spent executing by the CPU: 4862.00 millseconds
Time spent executing by the GPU events: 100.96 millseconds
Time spent executing by the GPU: 103.00 millseconds
Acceleration factor: 47.20
Relevance: true
Experiment for matrix size: 2048
