find_program(GLSLC
        glslc
        HINTS $ENV{VULKAN_SDK}/bin
        REQUIRED)

macro(add_shader source)
    add_custom_command(
            OUTPUT ${source}.spv
            DEPENDS ${source}
            COMMAND
            ${GLSLC} "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${CMAKE_CURRENT_SOURCE_DIR}/${source}.spv"
    )
endmacro()