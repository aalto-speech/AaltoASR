CXXFLAGS = -g -Wall -I../ -I/share/puhe/x86_64/include/lapackpp -I/share/puhe/x86_64/include/hcld/

OBJS = ../FeatureGenerator.o ../FeatureModules.o ../AudioReader.o \
	../ModuleConfig.o ../conf.o ../io.o ../str.o 

default: random_feature_test tests

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

random_feature_test: random_feature_test.o $(OBJS)
	$(CXX) -o $@ random_feature_test.o $(OBJS) -L/share/puhe/x86_64/lib -lfftw3 -lsndfile -lm -llapackpp -llapack -lhcld

.PHONY: tests
tests:
	sh run_tests.sh 2>&1 | tee log

.PHONY: clean
clean:
	rm -f random_feature_test{,.o} *.output log *.tmp *~
