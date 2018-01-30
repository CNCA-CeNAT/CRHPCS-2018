# test 1: matrix A times 0 matrix
OMP_NUM_THREADS=16 ./strassen 10 test_strassen_A.txt test_strassen_0.txt > result.txt 
./compare 100 result.txt result_strassen_A_0.txt
# test 2: matrix A times identity matrix
OMP_NUM_THREADS=16 ./strassen 10 test_strassen_A.txt test_strassen_I.txt > result.txt
./compare 100 result.txt result_strassen_A_I.txt
# test 3: identity matrix times identity matrix
OMP_NUM_THREADS=16 ./strassen 10 test_strassen_I.txt test_strassen_I.txt > result.txt
./compare 100 result.txt result_strassen_I_I.txt
# test 4: matrix A times matrix A
OMP_NUM_THREADS=16 ./strassen 10 test_strassen_A.txt test_strassen_A.txt > result.txt
./compare 100 result.txt result_strassen_A_A.txt
# removing result.txt file
rm -f result.txt
