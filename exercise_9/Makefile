.PHONY: build
build: ./bin/stencil

.PHONY: opt
opt: ./bin/stencil_opt

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/stencil: ./src/main.cu
	nvcc -O2 -arch=sm_20 --compiler-options "-O2 -Wextra" -o ./bin/stencil ./src/main.cu -I./inc/

./bin/stencil_opt: ./src/main.cu
	nvcc -O2 -arch=sm_20 --compiler-options "-O2 -Wextra" -D_OPT_KERNEL -o ./bin/stencil_opt ./src/main.cu -I./inc/
