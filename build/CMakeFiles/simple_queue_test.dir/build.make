# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /u/amargol/Masters/Robots/projects/cs393r_starter

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /u/amargol/Masters/Robots/projects/cs393r_starter/build

# Include any dependencies generated for this target.
include CMakeFiles/simple_queue_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/simple_queue_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/simple_queue_test.dir/flags.make

CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o: CMakeFiles/simple_queue_test.dir/flags.make
CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o: ../src/navigation/simple_queue_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/u/amargol/Masters/Robots/projects/cs393r_starter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o -c /u/amargol/Masters/Robots/projects/cs393r_starter/src/navigation/simple_queue_test.cc

CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /u/amargol/Masters/Robots/projects/cs393r_starter/src/navigation/simple_queue_test.cc > CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.i

CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /u/amargol/Masters/Robots/projects/cs393r_starter/src/navigation/simple_queue_test.cc -o CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.s

CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.requires:

.PHONY : CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.requires

CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.provides: CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.requires
	$(MAKE) -f CMakeFiles/simple_queue_test.dir/build.make CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.provides.build
.PHONY : CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.provides

CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.provides.build: CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o


# Object files for target simple_queue_test
simple_queue_test_OBJECTS = \
"CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o"

# External object files for target simple_queue_test
simple_queue_test_EXTERNAL_OBJECTS =

../bin/simple_queue_test: CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o
../bin/simple_queue_test: CMakeFiles/simple_queue_test.dir/build.make
../bin/simple_queue_test: CMakeFiles/simple_queue_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/u/amargol/Masters/Robots/projects/cs393r_starter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/simple_queue_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple_queue_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/simple_queue_test.dir/build: ../bin/simple_queue_test

.PHONY : CMakeFiles/simple_queue_test.dir/build

CMakeFiles/simple_queue_test.dir/requires: CMakeFiles/simple_queue_test.dir/src/navigation/simple_queue_test.cc.o.requires

.PHONY : CMakeFiles/simple_queue_test.dir/requires

CMakeFiles/simple_queue_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/simple_queue_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/simple_queue_test.dir/clean

CMakeFiles/simple_queue_test.dir/depend:
	cd /u/amargol/Masters/Robots/projects/cs393r_starter/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /u/amargol/Masters/Robots/projects/cs393r_starter /u/amargol/Masters/Robots/projects/cs393r_starter /u/amargol/Masters/Robots/projects/cs393r_starter/build /u/amargol/Masters/Robots/projects/cs393r_starter/build /u/amargol/Masters/Robots/projects/cs393r_starter/build/CMakeFiles/simple_queue_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/simple_queue_test.dir/depend

