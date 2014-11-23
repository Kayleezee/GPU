.PHONY: build
build: ./bin/memCpy

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/memCpy: ./src/main.cu
	nvcc -O2 --compiler-options "-O2 -Wextra" -o ./bin/memCpy ./src/main.cu -I./inc/ -arch sm_20

