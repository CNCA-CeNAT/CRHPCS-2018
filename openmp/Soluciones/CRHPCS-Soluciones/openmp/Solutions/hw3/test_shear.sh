#test 1: random sequence of integers 
OMP_NUM_THREADS=16 ./shear 100 test_shear_1.txt > result.txt
./compare 100 result.txt result_shear_1.txt
#test 2: inverted sequence
OMP_NUM_THREADS=16 ./shear 1024 test_shear_2.txt > result.txt
./compare 1024 result.txt result_shear_2.txt
#test 3: sequence of same number
OMP_NUM_THREADS=16 ./shear 289 test_shear_3.txt > result.txt
./compare 289 result.txt result_shear_3.txt
#test 4: random long sequence
OMP_NUM_THREADS=16 ./shear 10000 test_shear_4.txt > result.txt
./compare 10000 result.txt result_shear_4.txt
# removing result.txt file
rm -f result.txt
