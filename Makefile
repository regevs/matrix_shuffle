# CXX=clang++
CXX=g++

#CXXFLAGS=-Wall -O3 -DNDEBUG -I/usr/include/x86_64-linux-gnu/c++/4.8 -std=c++0x
#CXXFLAGS=-std=c++11 -Wall -pedantic -O3 -DNDEBUG -pthread -I/usr/local/include/eigen3/
#LDFLAGS=-O3 -pthread -lboost_system -lboost_thread-mt -lboost_program_options 

CXXFLAGS=-O3 -std=c++11  -Wall -pedantic -I/usr/local/include/eigen3/ -DNDEBUG -g
#CXXFLAGS=-O3 -std=c++11 -Wall -pedantic -I/home/yaelba/code/include/eigen3/eigen-eigen-323c052e1731 -DNDEBUG#  -Wcomment -DNDEBUG #-pg

# LDFLAGS=-O3 -lboost_system -lboost_program_options -lboost_filesystem #-pg
LDFLAGS=-O3 -lboost_system -lboost_program_options -lboost_filesystem

# DEBUG
#CXXFLAGS=-g -std=c++11  -Wall -pedantic -I/usr/local/include/eigen3/
#LDFLAGS=-g -lboost_system -lboost_program_options -lboost_filesystem

BIN=bin/matrix_shuffle_all
# BIN=test_stuff

SRC=$(wildcard src/matrix_shuffle_all.cpp)
# SRC=$(wildcard test_stuff.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) $(LDFLAGS) -o $(BIN) $^

%.o: %.c
	$(CXX) $@ -c $<

clean:
	rm -f bin/*.o
	rm $(BIN)


# g++ -std=c++11  -Wall -pedantic -DNDEBUG -O3 -lboost_system -lboost_program_options -lboost_filesystem -O3 -o fdd faster_discrete_distribution.cpp
# ./fdd