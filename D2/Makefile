COMPILERARGS=-std=c++1y -c -O3 -fopenmp
LINKERARGS=-fopenmp
INCLUDEDIRS=-I/usr/include/mpi -I../utils
LIBDIRS=-L/usr/lib/openmpi/lib -L../utils
LIBS=-lutils -lboost_system -lboost_filesystem -lmpi -lmpi_cxx
SRCDIR=src
GCC = g++

all: D2 DataLoader BinaryDataLoader TextDataLoader
	$(GCC) $(LIBDIRS) $(LINKERARGS) -o D2 D2.o DataLoader.o BinaryDataLoader.o TextDataLoader.o $(LIBS)

D2: $(SRCDIR)/D2.cpp
	$(GCC) $(COMPILERARGS) $(INCLUDEDIRS) $(SRCDIR)/D2.cpp

DataLoader: $(SRCDIR)/DataLoader.cpp
	$(GCC) $(COMPILERARGS) $(INCLUDEDIRS) $(SRCDIR)/DataLoader.cpp

BinaryDataLoader: $(SRCDIR)/BinaryDataLoader.cpp
	$(GCC) $(COMPILERARGS) $(INCLUDEDIRS) $(SRCDIR)/BinaryDataLoader.cpp

TextDataLoader: $(SRCDIR)/TextDataLoader.cpp
	$(GCC) $(COMPILERARGS) $(INCLUDEDIRS) $(SRCDIR)/TextDataLoader.cpp

clean:
	rm *.o D2

