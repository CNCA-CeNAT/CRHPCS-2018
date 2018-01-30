all: strassen shear

strassen:
	icpc -qopenmp strassen.cpp -o strassen
	icpc -w strassen.cpp -o strassen_serial
	
shear:
	icpc -qopenmp shear.cpp -o shear
	icpc -w shear.cpp -o shear_serial

compare:
	g++ compare.cpp -o compare

test: strassen shear compare
	./test_strassen.sh
	./test_shear.sh

clean:
	rm -f strassen strassen_serial shear shear_serial compare result.txt
