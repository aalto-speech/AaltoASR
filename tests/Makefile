CXXFLAGS = -g -Wall -I../

OBJS = ../FeatureGenerator.o ../FeatureModules.o ../AudioReader.o \
	../ModuleConfig.o ../conf.o ../io.o ../str.o 

default: random_feature_test tests

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

random_feature_test: random_feature_test.o $(OBJS)
	$(CXX) -o $@ random_feature_test.o $(OBJS) -lfftw3 -lsndfile

.PHONY: tests
tests:
	sh run_tests.sh 2>&1 | tee log

.PHONY: clean
clean:
	rm -f random_feature_test{,.o} *.output log *.tmp *~
