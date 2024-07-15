
CPPFLAGS:=$(shell pkg-config benchmark --cflags) $(shell pkg-config plumedInternals --cflags)
LDFLAGS=$(shell pkg-config benchmark --libs-only-l) $(shell pkg-config plumedInternals --libs) $(shell pkg-config plumedWrapper --libs) -lplumedKernel -pthread
CXXFLAGS=-O3 -std=c++17

%.o: %.cxx
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $<
exe: main.o
	$(CXX) $< -o $@ $(LDFLAGS)

#$(CXX) -std=c++17 main.cxx -o $@ `pkg-config benchmark --cflags` `pkg-config benchmark --libs` -lplumedWrapper -ldl -lplumed -fPIC -lplumedKernel 
	