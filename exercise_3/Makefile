#
# Exercise 3 - Makefile
# AUTHORS  Klaus Naumann
#          Alexander Schnapp
#          Günther Schindler
#
CC = nvcc
CFLAGS = -Wall

globalmemaccess :
	gcc $(CFLAGS)	src/part_1/globalmemaccess.cu
			-o matrixmultiply

clean :
	rm -f 	globalmemaccess

#.PHONY: build
#build: ./bin/memCpy

#.PHONY: clean
#clean:
#	rm ./bin/*
#	
#.PHONY: rebuild
#rebuild: clean build

#./bin/memCpy: ./src/main.cu
#	nvcc -O2 --compiler-options "-O2 -Wall -Wextra" -o ./bin/memCpy ./src/main.cu -I./inc/ -arch sm_20

