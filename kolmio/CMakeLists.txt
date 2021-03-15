add_shader(shaders/shader.vert)
add_shader(shaders/shader.frag)

corrade_add_resource(Shaders shaders.conf)

add_executable(kolmio
        main.cc
        ${Shaders})

target_include_directories(kolmio SYSTEM PRIVATE $ENV{VULKAN_SDK}/include)
target_link_libraries(kolmio
        ${Vulkan_LIBRARIES}
        glfw
        fmt::fmt
        Corrade::Utility)
