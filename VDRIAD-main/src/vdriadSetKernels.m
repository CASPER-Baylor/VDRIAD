% File:             vdriadSetKernels.m
% Author:           Jorge Augusto Martinez-Ortiz
% Date Created:     02.06.2023

function vdriadSetKernels(app)
%vdriadSetKernels Initializes the CUDA kernels that calculate the particles
%accelerations and positions
%   Detailed explanation goes here
    ThreadBlockSize = app.Parameters.BLOCK_SIZE;
    GridSize        = floor((app.Parameters.NUM_PARTICLES-1)/ThreadBlockSize) + 1;
    
    ! nvcc -ptx pcry.cu
    acc = parallel.gpu.CUDAKernel('pcry.ptx','pcry.cu','pcryCalculate_ACC');
    pos = parallel.gpu.CUDAKernel('pcry.ptx','pcry.cu','pcryCalculate_POS');
    
    acc.GridSize        = [GridSize 1 1];
    pos.GridSize        = [GridSize 1 1];
    acc.ThreadBlockSize = [ThreadBlockSize 1 1];
    pos.ThreadBlockSize = [ThreadBlockSize 1 1];

    app.pcryAcc = acc;
    app.pcryPos = pos;
end