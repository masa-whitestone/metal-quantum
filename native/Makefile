# Metal-Q Native Library Makefile

CC = clang
CFLAGS = -O3 -Wall -fPIC -fobjc-arc
LDFLAGS = -dynamiclib -framework Foundation -framework Metal

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# Files
SRCS = $(wildcard $(SRC_DIR)/*.m)
OBJS = $(SRCS:$(SRC_DIR)/%.m=$(BUILD_DIR)/%.o)
TARGET_LIB = $(BUILD_DIR)/libmetalq.dylib
METALLIB = $(BUILD_DIR)/default.metallib

# Copy of the build products inside the Python package, so wheel/editable
# installs ship working binaries (issue #7). The backend prefers native/build
# in a dev tree and falls back to metalq/lib; the native dladdr lookup finds
# default.metallib next to whichever dylib was loaded.
PYLIB_DIR = ../metalq/lib

# Targets
all: $(TARGET_LIB) $(METALLIB) pylib

pylib: $(TARGET_LIB) $(METALLIB)
	@mkdir -p $(PYLIB_DIR)
	cp -f $(TARGET_LIB) $(METALLIB) $(PYLIB_DIR)/

$(TARGET_LIB): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.m
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

$(METALLIB): $(SRC_DIR)/quantum_gates.metal $(SRC_DIR)/adjoint.metal
	@mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $(SRC_DIR)/quantum_gates.metal -o $(BUILD_DIR)/quantum_gates.air
	xcrun -sdk macosx metal -c $(SRC_DIR)/adjoint.metal -o $(BUILD_DIR)/adjoint.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/quantum_gates.air $(BUILD_DIR)/adjoint.air -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean pylib
