.PHONY: build
build: ./bin/reduction

.PHONY: clean
clean:
	rm -f ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/reduction: ./src/main.cu
	nvcc -arch=sm_20 -O2 --compiler-options "-O2 -Wextra" -o ./bin/reduction ./src/main.cu -I./inc/
