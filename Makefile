# Metal-Q Makefile

CC = clang
OBJC = clang
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

CFLAGS = -Wall -O2 -fPIC
OBJCFLAGS = -Wall -O2 -fPIC -fobjc-arc -framework Foundation -framework Metal

SRC_DIR = native
BUILD_DIR = build
PYTHON_DIR = metalq/python/metalq

# Source files
OBJC_SRCS = $(SRC_DIR)/metalq.m \
            $(SRC_DIR)/state_vector.m \
            $(SRC_DIR)/gate_executor.m \
            $(SRC_DIR)/measurement.m

METAL_SRC = $(SRC_DIR)/shaders/quantum_gates.metal

# Output
DYLIB = $(BUILD_DIR)/libmetalq.dylib
METALLIB_OUT = $(BUILD_DIR)/quantum_gates.metallib

.PHONY: all clean install test dirs

all: dirs $(DYLIB) $(METALLIB_OUT)

dirs:
	mkdir -p $(BUILD_DIR)

# Compile Metal shaders
$(BUILD_DIR)/quantum_gates.air: $(METAL_SRC)
	$(METAL) -c $< -o $@

$(METALLIB_OUT): $(BUILD_DIR)/quantum_gates.air
	$(METALLIB) $< -o $@

# Build dynamic library
$(DYLIB): $(OBJC_SRCS)
	$(OBJC) $(OBJCFLAGS) \
		-dynamiclib \
		-install_name @rpath/libmetalq.dylib \
		$(OBJC_SRCS) \
		-o $@

# Install to Python package
install: all
	mkdir -p $(PYTHON_DIR)/lib
	cp $(DYLIB) $(PYTHON_DIR)/lib/
	cp $(METALLIB_OUT) $(PYTHON_DIR)/lib/

# Run tests
test: install
	cd metalq/python && uv run pytest ../../tests/ -v

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(PYTHON_DIR)/lib
