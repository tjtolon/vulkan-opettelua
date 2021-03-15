cmake_minimum_required(VERSION 3.17)

project(vulkan-kokeilua LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(REGEX REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # use static c runtime
string(REGEX REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REGEX REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}") # use static c runtime
string(REGEX REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>") # use static c runtime

find_package(Vulkan REQUIRED) # Vulkan
find_package(fmt CONFIG REQUIRED)  # fmt::fmt
find_package(spdlog CONFIG REQUIRED) # spdlog::spdlog
find_package(glfw3 CONFIG REQUIRED) # glfw
find_package(Corrade REQUIRED Utility)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

include(cmake/vulkanstuff.cmake)

add_subdirectory(kolmio)

