#
# Escuela de veranillo en computacion de alto rendimiento.
# 24-28 julio 2017
# CNCA, CeNAT.
#
# M. Sc. Ricardo Roman-Brenes, PRIS-Lab/EIE/UCR
# ricardo.roman@ucr.ac.cr
#

SRC=darth.c
BIN=$(SRC).out

all:
	gcc-7 $(SRC) -o $(BIN)

# compilacion sin opciones
build0:
	gcc-7 $(SRC)

# especificando nombre del archivo de salida
build1:
	gcc-7 $(SRC) -o $(BIN)

# generando la tabla de simbolos para depuracion
build2:
	gcc-7 -g $(SRC) -o $(BIN)

# optimizacion nivel 0 (sin optimizacion)
O0:
	gcc-7 $(SRC) -O0 -o $(BIN).O0

# optimizacion nivel 3 (maximo nivel)
O3:
	gcc-7 $(SRC) -O3 -o $(BIN).O3

# solo compilar
compile_only:
	gcc-7 $(SRC) -C -o $(BIN).t

# solo enlazar
link_only:
	gcc-7 *.t -o $(BIN)

# generacion de codigo ensamblador
asm:
	gcc-7 $(SRC) -S

# generacion de codigo ensamblador con optimizacion 0
asm-O0:
	gcc-7 $(SRC) -O0 -S -o $(BIN).S.O0

# generacion de codigo ensamblador con optimizacion 3
asm-O3:
	gcc-7 $(SRC) -O3 -S -o $(BIN).S.O3
