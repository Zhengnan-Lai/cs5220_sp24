# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /global/common/software/nersc/pm-2021q4/sw/cmake-3.22.0/bin/cmake

# The command to remove a file.
RM = /global/common/software/nersc/pm-2021q4/sw/cmake-3.22.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/homes/z/zhengnan/cs5220_sp24/hw2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/homes/z/zhengnan/cs5220_sp24/hw2/build

# Include any dependencies generated for this target.
include CMakeFiles/serial.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/serial.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/serial.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/serial.dir/flags.make

CMakeFiles/serial.dir/main.cpp.o: CMakeFiles/serial.dir/flags.make
CMakeFiles/serial.dir/main.cpp.o: ../main.cpp
CMakeFiles/serial.dir/main.cpp.o: CMakeFiles/serial.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/z/zhengnan/cs5220_sp24/hw2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/serial.dir/main.cpp.o"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/serial.dir/main.cpp.o -MF CMakeFiles/serial.dir/main.cpp.o.d -o CMakeFiles/serial.dir/main.cpp.o -c /global/homes/z/zhengnan/cs5220_sp24/hw2/main.cpp

CMakeFiles/serial.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/serial.dir/main.cpp.i"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /global/homes/z/zhengnan/cs5220_sp24/hw2/main.cpp > CMakeFiles/serial.dir/main.cpp.i

CMakeFiles/serial.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/serial.dir/main.cpp.s"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /global/homes/z/zhengnan/cs5220_sp24/hw2/main.cpp -o CMakeFiles/serial.dir/main.cpp.s

CMakeFiles/serial.dir/serial.cpp.o: CMakeFiles/serial.dir/flags.make
CMakeFiles/serial.dir/serial.cpp.o: ../serial.cpp
CMakeFiles/serial.dir/serial.cpp.o: CMakeFiles/serial.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/z/zhengnan/cs5220_sp24/hw2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/serial.dir/serial.cpp.o"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/serial.dir/serial.cpp.o -MF CMakeFiles/serial.dir/serial.cpp.o.d -o CMakeFiles/serial.dir/serial.cpp.o -c /global/homes/z/zhengnan/cs5220_sp24/hw2/serial.cpp

CMakeFiles/serial.dir/serial.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/serial.dir/serial.cpp.i"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /global/homes/z/zhengnan/cs5220_sp24/hw2/serial.cpp > CMakeFiles/serial.dir/serial.cpp.i

CMakeFiles/serial.dir/serial.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/serial.dir/serial.cpp.s"
	/opt/cray/pe/craype/2.7.30/bin/CC $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /global/homes/z/zhengnan/cs5220_sp24/hw2/serial.cpp -o CMakeFiles/serial.dir/serial.cpp.s

# Object files for target serial
serial_OBJECTS = \
"CMakeFiles/serial.dir/main.cpp.o" \
"CMakeFiles/serial.dir/serial.cpp.o"

# External object files for target serial
serial_EXTERNAL_OBJECTS =

serial: CMakeFiles/serial.dir/main.cpp.o
serial: CMakeFiles/serial.dir/serial.cpp.o
serial: CMakeFiles/serial.dir/build.make
serial: CMakeFiles/serial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/z/zhengnan/cs5220_sp24/hw2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable serial"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/serial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/serial.dir/build: serial
.PHONY : CMakeFiles/serial.dir/build

CMakeFiles/serial.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/serial.dir/cmake_clean.cmake
.PHONY : CMakeFiles/serial.dir/clean

CMakeFiles/serial.dir/depend:
	cd /global/homes/z/zhengnan/cs5220_sp24/hw2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/z/zhengnan/cs5220_sp24/hw2 /global/homes/z/zhengnan/cs5220_sp24/hw2 /global/homes/z/zhengnan/cs5220_sp24/hw2/build /global/homes/z/zhengnan/cs5220_sp24/hw2/build /global/homes/z/zhengnan/cs5220_sp24/hw2/build/CMakeFiles/serial.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/serial.dir/depend
