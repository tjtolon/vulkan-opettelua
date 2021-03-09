add_executable(kolmio
        main.cc)

target_include_directories(kolmio SYSTEM PRIVATE $ENV{VULKAN_SDK}/include)
target_link_libraries(kolmio
        ${Vulkan_LIBRARIES}
        glfw
        fmt::fmt)
