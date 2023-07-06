CC = g++
CFLAGS = -Wall -Wextra -g -mavx2 --std=c++20


# Source files
SRCS = $(wildcard *.cpp)
# Header files
HDRS = $(wildcard *.h)
# Object files
OBJS = $(SRCS:.cpp=.o)
# Target name
TARGET = mpred

# Default target
all: $(TARGET)

# Rule to compile object files
%.o: %.cpp $(HDRS)
	$(CC) -c $(CFLAGS) $< -o $@

# Rule to link object files and generate the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Rule to clean the build files
clean:
	rm -f $(OBJS) $(TARGET)