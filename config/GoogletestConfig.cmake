message(STATUS "Including GoogletestConfig.cmake - building Gtest")


FetchContent_Declare(
  googletest
  URL          "${CMAKE_CURRENT_SOURCE_DIR}/vendor/googletest-03597a01ee50ed33e9dfd640b249b4be3799d395.zip"
  URL_HASH     SHA256=edd885a1ab32b6999515a880f669efadb80b3f880215f315985fa3f6eca7c4d3
)

# Force shared CRT for Windows MSVC
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(googletest)

enable_testing()