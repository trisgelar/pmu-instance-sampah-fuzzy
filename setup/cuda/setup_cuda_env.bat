@echo off
REM CUDA Environment Setup Script
REM Run this before using CUDA applications

set CUDA_HOME=D:\Programs\Nvidia\CUDA\v12.4
set CUDA_PATH=D:\Programs\Nvidia\CUDA\v12.4
set PATH=%CUDA_HOME%\bin;%PATH%

echo CUDA environment variables set:
echo CUDA_HOME=%CUDA_HOME%
echo CUDA_PATH=%CUDA_PATH%
echo.

REM Test CUDA
nvcc --version
echo.
nvidia-smi
echo.

REM Activate Python environment if needed
REM conda activate your_env_name

echo CUDA environment ready!
