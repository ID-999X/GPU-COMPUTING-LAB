# (NVCC) = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin\nvcc.exe"
# (NVCC_FLAGS)= -g -G -Xcompiler -Wall
# all: distanceMain.exe
# distanceMain.exe: distanceMain.o DistKernel.o
# 	$(NVCC) $^ -o $@
# distanceMain.o: distanceMain.cpp DistKernel.h
# 	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
# Distkernel.o: DistKernel.cu DistKernel.h
# 	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

# echo hello world

distanceMain.exe: distanceMain.obj DistKernel.obj
	nvcc distanceMain.obj DistKernel.obj
	./a.exe
# rm -rf a.exe
# rm a.lib
# rm a.exp
distanceMain.obj: distanceMain.cpp
	nvcc distanceMain.cpp -o distanceMain.obj -c
DistKernel.obj: DistKernel.cu DistKernel.h
	nvcc DistKernel.cu -o DistKernel.obj -c