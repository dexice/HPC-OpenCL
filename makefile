#Stéphane Küng
#Makefile
#10.12.2015
#High Performance Computing

OS:=$(shell uname)
CC=gcc
DST=opencl
SRC=$(wildcard *.cpp)
NVIDIA=$(wildcard /usr/lib/nvidia-*)
OBJ=$(SRC:.cpp=.o)

ifeq ($(OS),Linux)
	CC=g++
	LDFLAGS=-lm -lOpenCL #-L/usr/lib/nvidia-340
	CFLAGS=-O3 -Wall -I/usr/local/cuda/include
endif

ifeq ($(OS),Darwin)
	CC=g++
	LDFLAGS=-lm -framework OpenCL
	CFLAGS=-O2 -Wall
endif

all: $(DST)

rebuild: clean all

$(DST): $(OBJ)
	@echo Linking file : $@ from $^
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	@echo Compiling file : $@ from $<
	$(CC) $(CFLAGS) -o $@ -c $< 

clean:
	@echo Cleaning...
	@rm -rf *.o $(DST)

.PHONY: clean

run: $(DST)
	./$<