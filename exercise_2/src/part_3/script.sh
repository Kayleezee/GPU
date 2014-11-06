rm data.dat

for((i=1;i<1002;i=i+100))
	do	
	
		# programm , anzahl an KB, 1 for pageable memory, 1 for H2D
		./datamove $i 3 3 >>data.dat
	done

