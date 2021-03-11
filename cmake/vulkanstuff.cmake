find_program(GLSLC
        glslc
        HINTS $ENV{VULKAN_SDK}/bin
        REQUIRED)

find_program(GLSLANG_VALIDATOR
        glslangValidator
        HINTS $ENV{VULKAN_SDK}/bin
        REQUIRED)

find_program(SPIRV_VAL
        spirv-val
        HINTS $ENV{VULKAN_SDK}/bin
        REQUIRED)

find_program(SPIRV_OPT
        spirv-opt
        HINTS $ENV{VULKAN_SDK}/bin
        REQUIRED)

find_program(SPIRV_CROSS
        spirv-cross
        HINTS $ENV{VULKAN_SDK}/bin
        REQUIRED)
