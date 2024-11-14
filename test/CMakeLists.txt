enable_testing()
include(CTest)

add_library(testutils testutils.cpp testutils.h)
target_link_libraries(testutils PUBLIC fitlib)

add_executable(test_serial test_serial.cpp)
target_link_libraries(test_serial PRIVATE fitlib Catch2::Catch2WithMain)

add_executable(test_cuda test_cuda.cpp)
target_link_libraries(test_cuda PRIVATE fitlib Catch2::Catch2WithMain)


