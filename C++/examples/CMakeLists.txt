# CPL-Sync command-line driver
add_executable(CPL-Sync cpl_sync.cpp)
target_link_libraries(CPL-Sync SESync CPLSync)

if(${ENABLE_PROFILING})
target_compile_definitions(CPL-Sync PRIVATE GPERFTOOLS)
target_link_libraries(CPL-Sync profiler)
endif()

if(${ENABLE_OPENMP})
if(OPENMP_FOUND)
set_target_properties(CPL-Sync PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif()
endif()

message(STATUS "Building main CPL-Sync command-line executable in directory ${EXECUTABLE_OUTPUT_PATH}\n")
