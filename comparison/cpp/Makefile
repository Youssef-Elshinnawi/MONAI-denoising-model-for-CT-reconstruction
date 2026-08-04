CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude

SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, build/%.o, $(SRC))
TARGET = build/iterative_recon

OBJ_NO_MAIN = $(filter-out build/main.o, $(OBJ))
OBJ_NO_CT_COMPARE = $(filter-out build/main_ct_compare.o, $(OBJ))
CT_COMPARE_TARGET = build/ct_compare

all: $(TARGET)

$(TARGET): $(OBJ_NO_CT_COMPARE)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ_NO_CT_COMPARE)

$(CT_COMPARE_TARGET): build/main_ct_compare.o $(OBJ_NO_MAIN)
	$(CXX) $(CXXFLAGS) -o $@ $^

build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f build/*.o $(TARGET) $(CT_COMPARE_TARGET)

.PHONY: all clean
