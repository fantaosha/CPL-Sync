# CPL-Sync command-line driver
add_executable(CPL-Sync cpl_sync.cpp)
target_link_libraries(CPL-Sync SESync CPLSync)
if(${ENABLE_PROFILING})
target_compile_definitions(SE-Sync PRIVATE GPERFTOOLS)
target_link_libraries(SE-Sync profiler)
endif()


message(STATUS "Building main SE-Sync command-line executable in directory ${EXECUTABLE_OUTPUT_PATH}\n")
