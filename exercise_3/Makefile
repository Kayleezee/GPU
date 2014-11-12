#
# Exercise 3 - Makefile
# AUTHORS  Klaus Naumann
#          Alexander Schnapp
#          Günther Schindler
#
CC = nvcc
CFLAGS =

globalmemaccess :
	$(CC) -o globalmemaccess src/part_1/globalmemaccess.cu

memCpy :
	$(CC) -O2 --compiler-options "-O2 -Wall -Wextra" -o memCpy ./src/main.cu -I./inc/ -arch sm_20

clean :
	rm -f globalmemaccess memCpy
