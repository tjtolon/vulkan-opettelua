#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

#include <vulkan/vulkan.hpp>
#include <spdlog/spdlog.h>

#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window;
    vk::Instance instance;
    const char* validation_layers[1] = {"VK_LAYER_KHRONOS_validation"};

    void initWindow() {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        checkValidationLayerSupport();
        uint32_t glfw_extension_count = 0;
        const char **glfw_extension{};
        glfw_extension = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
        vk::ApplicationInfo applicationInfo{
                .pApplicationName = "Kolmio",
                .applicationVersion = 1,
                .pEngineName = "Esimerkkimoottori",
                .engineVersion = 1,
                .apiVersion = VK_API_VERSION_1_2
        };

        vk::InstanceCreateInfo create_info{
                .pApplicationInfo = &applicationInfo,
                .enabledLayerCount = 1,
                .ppEnabledLayerNames = validation_layers,
                .enabledExtensionCount = glfw_extension_count,
                .ppEnabledExtensionNames = glfw_extension,
                };
        try {
            instance = vk::createInstance(create_info);
        } catch (vk::SystemError &err) {
            throw std::runtime_error(fmt::format("failed to create instance! err: {}", err.what()));
        }
        std::vector<vk::ExtensionProperties> extension_properties = vk::enumerateInstanceExtensionProperties();
        spdlog::info("Available extensions:");
        for (auto &extension : extension_properties) {
            spdlog::info("\t{}", extension.extensionName);
        }
    }

    void mainLoop() {

    }

    void cleanup() {
        instance.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    bool checkValidationLayerSupport() {
        bool all_ok = true;
        std::vector<vk::LayerProperties> available_layers = vk::enumerateInstanceLayerProperties();
        spdlog::info("Available layers:");
        for (auto &available_layer : available_layers) {
            spdlog::info("\t{}", available_layer.layerName);
        }
        for (const auto& required_layer_name : validation_layers) {
            bool found = false;
            for (auto &available_layer : available_layers) {
                if (std::strcmp(available_layer.layerName, required_layer_name) == 0) {
                    found = true;
                }
            }
            if (!found) {
                all_ok = false;
                spdlog::error("Could not found the required layer {}", required_layer_name);
            }
        }
        return all_ok;
    }
};

int main(int argc, char **argv) {
    HelloTriangleApplication app;
    try {
        app.run();
    } catch (const std::exception &e) {
        spdlog::log(spdlog::level::err, "{}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}