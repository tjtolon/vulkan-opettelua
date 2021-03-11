#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

#include <vulkan/vulkan.hpp>
#include <spdlog/spdlog.h>
#include <optional>
#include <set>

#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#include <cstdint>
#include <algorithm>

PFN_vkCreateDebugUtilsMessengerEXT pfnVkCreateDebugUtilsMessengerEXT;
PFN_vkDestroyDebugUtilsMessengerEXT pfnVkDestroyDebugUtilsMessengerEXT;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(VkInstance instance,
                                                              const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                                              const VkAllocationCallbacks* pAllocator,
                                                              VkDebugUtilsMessengerEXT* pMessenger) {
    return pfnVkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(VkInstance instance,
                                                           VkDebugUtilsMessengerEXT messenger,
                                                           VkAllocationCallbacks const* pAllocator) {
    return pfnVkDestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator);
}

const std::vector<const char*> device_extensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> format;
    std::vector<vk::PresentModeKHR> present_modes;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    bool enable_validation_layers{false};
    bool print_available_extensions_and_layers{false};
    GLFWwindow* window;
    vk::Instance instance;
    const char* validation_layers[1] = {"VK_LAYER_KHRONOS_validation"};
    vk::DebugUtilsMessengerCreateInfoEXT debug_create_info;
    vk::DebugUtilsMessengerEXT debug_utils_messenger;
    vk::PhysicalDevice physical_device;
    vk::Device logical_device;
    vk::Queue graphics_queue;
    vk::SurfaceKHR surface;
    vk::Queue present_queue;

    vk::SwapchainKHR swap_chain;
    std::vector<vk::Image> swap_chain_images;
    vk::Format swap_chain_image_format;
    vk::Extent2D swap_chain_extent;
    std::vector<vk::ImageView> swap_chain_image_views;

    void initWindow() {
        if (!glfwInit()) {
            throw std::runtime_error("failed to init glfw");
        }
        if (!glfwVulkanSupported()) {
            throw std::runtime_error("vulkan not supported (glfw check)");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
#ifndef NDEBUG
        enable_validation_layers = true;
        print_available_extensions_and_layers = true;
#endif
        if (print_available_extensions_and_layers) {
            printAvailableFunctionality();
        }
        checkValidationLayerSupport();
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createGraphicsPipeline();
    }

    void mainLoop() {

    }

    void cleanup() {
        for (const auto& image_view : swap_chain_image_views) {
            logical_device.destroy(image_view);
        }
        logical_device.destroy(swap_chain);
        logical_device.destroy();
        if (enable_validation_layers) {
            instance.destroyDebugUtilsMessengerEXT(debug_utils_messenger);
        }
        instance.destroySurfaceKHR(surface);
        instance.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createInstance() {
        vk::ApplicationInfo applicationInfo{
                .pApplicationName = "Kolmio",
                .applicationVersion = 1,
                .pEngineName = "Esimerkkimoottori",
                .engineVersion = 1,
                .apiVersion = VK_API_VERSION_1_2
        };

        auto extensions = getRequiredExtensions();

        vk::InstanceCreateInfo create_info{
                .pApplicationInfo = &applicationInfo,
                .enabledLayerCount = 1,
                .ppEnabledLayerNames = validation_layers,
                .enabledExtensionCount = uint32_t(extensions.size()),
                .ppEnabledExtensionNames = extensions.data(),
        };
        if (enable_validation_layers) {
            create_info.pNext = &debug_create_info;
        }
        try {
            instance = vk::createInstance(create_info);
            if (enable_validation_layers) {
                pfnVkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                        instance.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
                if (!pfnVkCreateDebugUtilsMessengerEXT) {
                    throw std::runtime_error("Unable to find  pfnCreateDebugUtilsMessengerEXT function");
                }

                pfnVkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                        instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
                if (!pfnVkDestroyDebugUtilsMessengerEXT) {
                    throw std::runtime_error("Unable to find  pfnVkDestroyDebugUtilsMessengerEXT function");
                }
            }

        } catch (vk::SystemError& err) {
            throw std::runtime_error(fmt::format("failed to create instance! err: {}", err.what()));
        }

    }

    void setupDebugMessenger() {
        if (!enable_validation_layers) return;
        debug_create_info = {
                .sType = vk::DebugUtilsMessengerCreateInfoEXT::structureType,
                .messageSeverity = {vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError},
                .messageType = {vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance},
                .pfnUserCallback = debugCallback,
                .pUserData = nullptr
        };
        debug_utils_messenger = instance.createDebugUtilsMessengerEXT(debug_create_info);
    }

    void pickPhysicalDevice() {
        bool physical_device_found = false;
        std::vector<vk::PhysicalDevice> physical_devices = instance.enumeratePhysicalDevices();
        if (physical_devices.empty()) {
            throw std::runtime_error("Could not find a GPU with Vulkan support");
        }
        if (print_available_extensions_and_layers) {
            spdlog::info("Available devices:");
            for (const auto& d : physical_devices) {
                const auto& properties = d.getProperties();
                spdlog::info("\t{}", properties.deviceName);
                printPhysicalDeviceFunctionality("\t", d);
            }
        }
        for (auto& d : physical_devices) {
            if (isDeviceSuitable(d)) {
                physical_device = d;
                physical_device_found = true;
                break;
            }
        }
        if (!physical_device_found) {
            throw std::runtime_error("failed to find a suitable GPU");
        }


    }

    bool isDeviceSuitable(const vk::PhysicalDevice& p_dev) {
        const auto& properties = p_dev.getProperties();
        const auto& features = p_dev.getFeatures();
        QueueFamilyIndices indices = findQueueFamilies(p_dev);
        bool extensions_supported = checkDeviceExtensionSupport(p_dev);
        bool swap_chain_is_adequate = false;
        if (extensions_supported) {
            auto swap_chain_support = querySwapChainSupport(p_dev);
            swap_chain_is_adequate = !swap_chain_support.format.empty() && !swap_chain_support.present_modes.empty();
        }
        return indices.isComplete() && extensions_supported && swap_chain_is_adequate;
    }

    bool checkValidationLayerSupport() {
        bool all_ok = true;
        std::vector<vk::LayerProperties> available_layers = vk::enumerateInstanceLayerProperties();
        for (const auto& required_layer_name : validation_layers) {
            bool found = false;
            for (auto& available_layer : available_layers) {
                if (std::strcmp(available_layer.layerName, required_layer_name) == 0) {
                    found = true;
                }
            }
            if (!found) {
                all_ok = false;
                spdlog::error("Could not find the required layer {}", required_layer_name);
            }
        }
        return all_ok;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfw_extension_count = 0;
        const char** glfw_extension{};
        glfw_extension = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
        std::vector<const char*> extensions(glfw_extension, glfw_extension + glfw_extension_count);
        if (enable_validation_layers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
            VkDebugUtilsMessageTypeFlagsEXT message_type,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) {
        if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
            spdlog::info("validation layer: {}", pCallbackData->pMessage);
        } else if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
            spdlog::debug("validation layer: {}", pCallbackData->pMessage);
        } else if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            spdlog::warn("validation layer: {}", pCallbackData->pMessage);
        } else if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            spdlog::error("validation layer: {}", pCallbackData->pMessage);
        } else {
            spdlog::error("WEIRD SEVERITY! validation layer: {}", pCallbackData->pMessage);
        }
        return VK_FALSE;
    }

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& p_dev) {
        QueueFamilyIndices indices;
        std::vector<vk::QueueFamilyProperties> queue_families = p_dev.getQueueFamilyProperties();
        int i = 0;
        for (const auto& queue_family : queue_families) {
            if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }
            if (p_dev.getSurfaceSupportKHR(i, surface)) {
                indices.presentFamily = i;
            }
            ++i;
        }
        return indices;
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physical_device);
        std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> unique_queue_families = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        float queue_priority = 1.0f;
        for (auto queue_family : unique_queue_families) {
            vk::DeviceQueueCreateInfo queue_create_info{
                    .sType = vk::DeviceQueueCreateInfo::structureType,
                    .queueFamilyIndex = queue_family,
                    .queueCount = 1,
                    .pQueuePriorities = &queue_priority,
            };
            queue_create_infos.push_back(queue_create_info);
        }

        vk::PhysicalDeviceFeatures features{

        };
        vk::DeviceCreateInfo device_create_info{
                .sType = vk::DeviceCreateInfo::structureType,
                .queueCreateInfoCount = uint32_t(queue_create_infos.size()),
                .pQueueCreateInfos = queue_create_infos.data(),
                .enabledExtensionCount = uint32_t(device_extensions.size()),
                .ppEnabledExtensionNames = device_extensions.data(),
                .pEnabledFeatures = &features,
        };
        if (enable_validation_layers) {
            device_create_info.enabledLayerCount = 1;
            device_create_info.ppEnabledLayerNames = validation_layers;
        } else {
            device_create_info.enabledLayerCount = 0;
        }
        logical_device = physical_device.createDevice(device_create_info);
        graphics_queue = logical_device.getQueue(indices.graphicsFamily.value(), 0);
        present_queue = logical_device.getQueue(indices.presentFamily.value(), 0);

    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, (VkSurfaceKHR*) &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface");
        };
    }

    static bool checkDeviceExtensionSupport(const vk::PhysicalDevice& p_dev) {
        std::vector<vk::ExtensionProperties> extensions_supported = p_dev.enumerateDeviceExtensionProperties();
        std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());
        for (const auto& extension : extensions_supported) {
            required_extensions.erase(extension.extensionName);
        }
        return required_extensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& p_dev) {
        SwapChainSupportDetails details;
        details.capabilities = p_dev.getSurfaceCapabilitiesKHR(surface);
        details.format = p_dev.getSurfaceFormatsKHR(surface);
        details.present_modes = p_dev.getSurfacePresentModesKHR(surface);
        return details;
    }

    void printAvailableFunctionality() {
        std::vector<vk::ExtensionProperties> extension_properties = vk::enumerateInstanceExtensionProperties();
        spdlog::info("Available extensions:");
        for (auto& extension : extension_properties) {
            spdlog::info("\t{}", extension.extensionName);
        }
        std::vector<vk::LayerProperties> available_layers = vk::enumerateInstanceLayerProperties();
        spdlog::info("Available layers:");
        for (auto& available_layer : available_layers) {
            spdlog::info("\t{}", available_layer.layerName);
        }

    }

    void printPhysicalDeviceFunctionality(const std::string& prefix, const vk::PhysicalDevice& p_dev) {
        std::vector<vk::QueueFamilyProperties> queue_families = p_dev.getQueueFamilyProperties();
        spdlog::info("{}Available Queue Families:", prefix);
        for (const auto& queue_family : queue_families) {
            if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
                spdlog::info("{}\tGraphics", prefix);
            }
            if (queue_family.queueFlags & vk::QueueFlagBits::eCompute) {
                spdlog::info("{}\tCompute", prefix);
            }
            if (queue_family.queueFlags & vk::QueueFlagBits::eProtected) {
                spdlog::info("{}\tProtected", prefix);
            }
            if (queue_family.queueFlags & vk::QueueFlagBits::eSparseBinding) {
                spdlog::info("{}\tSparse Binding", prefix);
            }
            if (queue_family.queueFlags & vk::QueueFlagBits::eTransfer) {
                spdlog::info("{}\tTransfer", prefix);
            }
        }
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& available_formats) {
        for (const auto& available_format : available_formats) {
            if (available_format.format == vk::Format::eB8G8R8A8Srgb &&
                available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return available_format;
            }
        }
        return available_formats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& available_present_modes) {
        for (const auto& available_present_mode : available_present_modes) {
            if (available_present_mode == vk::PresentModeKHR::eMailbox) {
                return available_present_mode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        } else {
            int width{};
            int height{};
            glfwGetFramebufferSize(window, &width, &height);
            vk::Extent2D actual_extent{
                    .width = uint32_t(width),
                    .height = uint32_t(height)
            };

            actual_extent.width = std::max(capabilities.minImageExtent.width,
                                           std::min(capabilities.maxImageExtent.width, actual_extent.width));
            actual_extent.height = std::max(capabilities.minImageExtent.height,
                                            std::min(capabilities.maxImageExtent.height, actual_extent.height));
            return actual_extent;
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swap_chain_support = querySwapChainSupport(physical_device);
        vk::SurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.format);
        vk::PresentModeKHR present_mode = chooseSwapPresentMode(swap_chain_support.present_modes);
        vk::Extent2D extent = chooseSwapExtent(swap_chain_support.capabilities);

        uint32_t image_count = std::min(swap_chain_support.capabilities.minImageCount + 1,
                                        swap_chain_support.capabilities.maxImageCount);

        vk::SwapchainCreateInfoKHR create_info{
                .sType = vk::SwapchainCreateInfoKHR::structureType,
                .surface = surface,
                .minImageCount = image_count,
                .imageFormat = surface_format.format,
                .imageColorSpace = surface_format.colorSpace,
                .imageExtent = extent,
                .imageArrayLayers = 1,
                .imageUsage = vk::ImageUsageFlagBits::eColorAttachment
        };

        auto indices = findQueueFamilies(physical_device);
        uint32_t queue_family_indices[2] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        if (indices.graphicsFamily != indices.presentFamily) {
            create_info.imageSharingMode = vk::SharingMode::eConcurrent;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queue_family_indices;
        } else {
            create_info.imageSharingMode = vk::SharingMode::eExclusive;
            create_info.queueFamilyIndexCount = 0;
            create_info.pQueueFamilyIndices = nullptr;
        }

        create_info.preTransform = swap_chain_support.capabilities.currentTransform;
        create_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        create_info.oldSwapchain = nullptr;

        swap_chain = logical_device.createSwapchainKHR(create_info);
        swap_chain_images = logical_device.getSwapchainImagesKHR(swap_chain);
        swap_chain_image_format = surface_format.format;
        swap_chain_extent = extent;
    }

    void createImageViews() {
        swap_chain_image_views.resize(swap_chain_images.size());
        for (size_t i = 0; i < swap_chain_images.size(); ++i) {
            vk::ImageViewCreateInfo create_info{
                    .sType = vk::ImageViewCreateInfo::structureType,
                    .image = swap_chain_images[i],
                    .viewType = vk::ImageViewType::e2D,
                    .format = swap_chain_image_format,
                    .components = {
                            .r = vk::ComponentSwizzle::eIdentity,
                            .g = vk::ComponentSwizzle::eIdentity,
                            .b = vk::ComponentSwizzle::eIdentity,
                            .a = vk::ComponentSwizzle::eIdentity,
                    },
                    .subresourceRange = {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                    }
            };
            swap_chain_image_views[i] = logical_device.createImageView(create_info);
        }
    }

    void createGraphicsPipeline() {

    }
};

int main(int argc, char** argv) {
    HelloTriangleApplication app;
    try {
        app.run();
    } catch (const std::exception& e) {
        spdlog::log(spdlog::level::err, "{}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}