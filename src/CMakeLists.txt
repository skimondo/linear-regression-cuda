add_library(fitlib STATIC

  experiments.cpp
  experiments.h

  fitserial.cpp
  fitserial.h

  fitcuda.cu
  fitcuda.h

  optparser.cpp
  optparser.hpp

)
target_include_directories(fitlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fitab main.cpp)
target_link_libraries(fitab PRIVATE fitlib)
