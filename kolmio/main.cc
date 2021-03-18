#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

#include <vulkan/vulkan.hpp>
#include <spdlog/spdlog.h>
#include <optional>
#include <set>
#include "utils.hh"
#include <Corrade/Utility/Resource.h>

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

const int MAX_FRAMES_IN_FLIGHT = 2;

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
    std::vector<vk::Framebuffer> swap_chain_framebuffers;

    vk::PipelineLayout pipeline_layout;
    vk::RenderPass render_pass;
    vk::Pipeline graphics_pipeline;

    vk::CommandPool command_pool;
    std::vector<vk::CommandBuffer> command_buffers;

    std::vector<vk::Semaphore> image_available_semaphores;
    std::vector<vk::Semaphore> render_finished_semaphores;
    std::vector<vk::Fence> in_flight_fences;
    std::vector<vk::Fence> images_in_flight;
    size_t current_frame = 0;

    bool frame_buffer_resized = false;


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
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }
    static void framebufferResizeCallback(GLFWwindow* window, int, int) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->frame_buffer_resized = true;
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
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        logical_device.waitIdle();
    }

    void cleanupSwapChain() {
        for (auto framebuffer : swap_chain_framebuffers) {
            logical_device.destroy(framebuffer);
        }
        logical_device.freeCommandBuffers(command_pool, command_buffers);
        logical_device.destroy(graphics_pipeline);
        logical_device.destroy(pipeline_layout);
        logical_device.destroy(render_pass);
        for (const auto& image_view : swap_chain_image_views) {
            logical_device.destroy(image_view);
        }
        logical_device.destroy(swap_chain);
    }

    void cleanup() {
        cleanupSwapChain();
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            logical_device.destroy(render_finished_semaphores[i]);
            logical_device.destroy(image_available_semaphores[i]);
            logical_device.destroy(in_flight_fences[i]);
        }
        logical_device.destroy(command_pool);
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
//        const auto& properties = p_dev.getProperties();
//        const auto& features = p_dev.getFeatures();
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
            VkDebugUtilsMessageTypeFlagsEXT /*message_type*/,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* /*pUserData*/) {
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
        if (glfwCreateWindowSurface(instance, window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS) {
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
        auto vert_shader_module = createShaderModule("shaders/shader.vert.spv");
        auto frag_shader_module = createShaderModule("shaders/shader.frag.spv");

        vk::PipelineShaderStageCreateInfo vert_stage_info{
                .sType = vk::PipelineShaderStageCreateInfo::structureType,
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = vert_shader_module,
                .pName = "main"
        };

        vk::PipelineShaderStageCreateInfo frag_stage_info{
                .sType = vk::PipelineShaderStageCreateInfo::structureType,
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = frag_shader_module,
                .pName = "main"
        };
        vk::PipelineShaderStageCreateInfo stages[2] = {vert_stage_info, frag_stage_info};

        vk::PipelineVertexInputStateCreateInfo input_stage_info{
                .sType = vk::PipelineVertexInputStateCreateInfo::structureType,
                .vertexBindingDescriptionCount = 0,
                .pVertexBindingDescriptions = nullptr,
                .vertexAttributeDescriptionCount = 0,
                .pVertexAttributeDescriptions = nullptr,
        };

        vk::PipelineInputAssemblyStateCreateInfo input_assembly_stage_ifno{
                .sType = vk::PipelineInputAssemblyStateCreateInfo::structureType,
                .topology = vk::PrimitiveTopology::eTriangleList,
                .primitiveRestartEnable = VK_FALSE,
        };

        vk::Viewport viewport{
                .x = 0.f,
                .y = 0.f,
                .width = float(swap_chain_extent.width),
                .height = float(swap_chain_extent.height),
                .minDepth = 0.f,
                .maxDepth = 1.f,
        };

        vk::Rect2D scissor{
                .offset = {0, 0},
                .extent = swap_chain_extent,
        };

        vk::PipelineViewportStateCreateInfo viewport_state{
                .sType = vk::PipelineViewportStateCreateInfo::structureType,
                .viewportCount = 1,
                .pViewports = &viewport,
                .scissorCount = 1,
                .pScissors = &scissor,
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eBack,
                .frontFace = vk::FrontFace::eClockwise,
                .depthBiasEnable = VK_FALSE,
                .depthBiasConstantFactor = 0.f,
                .depthBiasClamp = 0.f,
                .depthBiasSlopeFactor = 0.f,
                .lineWidth = 1.f,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
                .sType = vk::PipelineMultisampleStateCreateInfo::structureType,
                .rasterizationSamples = vk::SampleCountFlagBits::e1,
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = 1.f,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE,
        };

        vk::PipelineColorBlendAttachmentState color_blend_attachment{
                .blendEnable = VK_FALSE,
                .srcColorBlendFactor = vk::BlendFactor::eOne,
                .dstColorBlendFactor = vk::BlendFactor::eZero,
                .colorBlendOp = vk::BlendOp::eAdd,
                .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                .dstAlphaBlendFactor = vk::BlendFactor::eZero,
                .alphaBlendOp = vk::BlendOp::eAdd,
                .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        vk::ArrayWrapper1D<float, 4> w;
        vk::PipelineColorBlendStateCreateInfo color_blending{
                .sType = vk::PipelineColorBlendStateCreateInfo::structureType,
                .logicOpEnable = VK_FALSE,
                .logicOp = vk::LogicOp::eCopy,
                .attachmentCount = 1,
                .pAttachments = &color_blend_attachment,
                .blendConstants = std::array<float, 4>{0.f, 0.f, 0.f, 0.f},
        };

        vk::PipelineLayoutCreateInfo pipeline_layout_info{
                .sType = vk::PipelineLayoutCreateInfo::structureType,
                .setLayoutCount = 0,
                .pSetLayouts = nullptr,
                .pushConstantRangeCount = 0,
                .pPushConstantRanges = nullptr,
        };

        pipeline_layout = logical_device.createPipelineLayout(pipeline_layout_info);

        vk::GraphicsPipelineCreateInfo create_info{
                .sType = vk::GraphicsPipelineCreateInfo::structureType,
                .stageCount = 2,
                .pStages = stages,
                .pVertexInputState = &input_stage_info,
                .pInputAssemblyState = &input_assembly_stage_ifno,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = nullptr,
                .pColorBlendState = &color_blending,
                .pDynamicState = nullptr,
                .layout = pipeline_layout,
                .renderPass = render_pass,
                .subpass = 0,
                .basePipelineHandle = nullptr,
                .basePipelineIndex = -1
        };
        vk::Result r;
        std::tie(r, graphics_pipeline) = logical_device.createGraphicsPipeline(vk::PipelineCache{}, create_info);


        logical_device.destroy(vert_shader_module);
        logical_device.destroy(frag_shader_module);
    }


    vk::ShaderModule createShaderModule(const std::string& resource_name) {
        Corrade::Utility::Resource rs("shaders");
        auto spirv = rs.getRaw(resource_name);
        vk::ShaderModuleCreateInfo create_info{
                .sType = vk::ShaderModuleCreateInfo::structureType,
                .codeSize = spirv.size(),
                .pCode = reinterpret_cast<const uint32_t*>(spirv.data()),
        };
        return logical_device.createShaderModule(create_info);

    }

    void createRenderPass() {
        vk::AttachmentDescription color_attachment{
                .format = swap_chain_image_format,
                .samples = vk::SampleCountFlagBits::e1,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eUndefined,
                .finalLayout = vk::ImageLayout::ePresentSrcKHR,
        };
        vk::AttachmentReference color_attachment_ref{
                .attachment = 0,
                .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        vk::SubpassDescription subpass{
                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                .colorAttachmentCount = 1,
                .pColorAttachments = &color_attachment_ref,
        };

        vk::SubpassDependency dependency{
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = {vk::PipelineStageFlagBits::eColorAttachmentOutput},
                .dstStageMask = {vk::PipelineStageFlagBits::eColorAttachmentOutput},
                .dstAccessMask = {vk::AccessFlagBits::eColorAttachmentWrite},
        };

        vk::RenderPassCreateInfo create_info{
                .sType = vk::RenderPassCreateInfo::structureType,
                .attachmentCount = 1,
                .pAttachments = &color_attachment,
                .subpassCount = 1,
                .pSubpasses = &subpass,
                .dependencyCount = 1,
                .pDependencies = &dependency,
        };

        render_pass = logical_device.createRenderPass(create_info);
    }

    void createFramebuffers() {
        swap_chain_framebuffers.resize(swap_chain_image_views.size());
        for (size_t i = 0; i < swap_chain_image_views.size(); ++i) {
            vk::ImageView attachments[1]{
                    swap_chain_image_views[i]
            };

            vk::FramebufferCreateInfo create_info{
                    .sType = vk::FramebufferCreateInfo::structureType,
                    .renderPass = render_pass,
                    .attachmentCount = 1,
                    .pAttachments =  attachments,
                    .width = swap_chain_extent.width,
                    .height = swap_chain_extent.height,
                    .layers = 1,
            };

            swap_chain_framebuffers[i] = logical_device.createFramebuffer(create_info);
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queue_family_indices = findQueueFamilies(physical_device);

        vk::CommandPoolCreateInfo create_info{
                .sType = vk::CommandPoolCreateInfo::structureType,
                .queueFamilyIndex = queue_family_indices.graphicsFamily.value(),
        };
        command_pool = logical_device.createCommandPool(create_info);
    }

    void createCommandBuffers() {
        command_buffers.resize(swap_chain_framebuffers.size());

        vk::CommandBufferAllocateInfo allocate_info{
                .sType = vk::CommandBufferAllocateInfo::structureType,
                .commandPool = command_pool,
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = uint32_t(command_buffers.size()),
        };

        command_buffers = logical_device.allocateCommandBuffers(allocate_info);

        for (size_t i = 0; i < command_buffers.size(); ++i) {
            vk::CommandBufferBeginInfo begin_info{
                    .sType = vk::CommandBufferBeginInfo::structureType,
            };
            command_buffers[i].begin(begin_info);

            vk::ClearValue clear_color{};
            clear_color.color = std::array<float, 4>{0.f, 0.f, 0.f, 1.f};
            vk::RenderPassBeginInfo render_pass_info{
                    .sType = vk::RenderPassBeginInfo::structureType,
                    .renderPass = render_pass,
                    .framebuffer = swap_chain_framebuffers[i],
                    .renderArea = {
                            .offset = {0, 0},
                            .extent = swap_chain_extent,
                    },
                    .clearValueCount = 1,
                    .pClearValues = &clear_color,
            };
            command_buffers[i].beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

            command_buffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline);

            command_buffers[i].draw(3, 1, 0, 0);

            command_buffers[i].endRenderPass();

            command_buffers[i].end();
        }
    }

    void createSyncObjects() {
        image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
        images_in_flight.resize(swap_chain_images.size(), nullptr);

        vk::SemaphoreCreateInfo create_info{
                .sType = vk::SemaphoreCreateInfo::structureType,
        };
        vk::FenceCreateInfo fence_info{
                .sType = vk::FenceCreateInfo::structureType,
                .flags = vk::FenceCreateFlagBits::eSignaled,
        };
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            image_available_semaphores[i] = logical_device.createSemaphore(create_info);
            render_finished_semaphores[i] = logical_device.createSemaphore(create_info);
            in_flight_fences[i] = logical_device.createFence(fence_info);

        }
    }

    void drawFrame() {
        while (vk::Result::eTimeout ==
               logical_device.waitForFences(1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX));
        uint32_t image_index{};

        auto result = logical_device.acquireNextImageKHR(swap_chain, UINT64_MAX,
                                                         image_available_semaphores[current_frame], nullptr,
                                                         &image_index);
        switch (result) {
            case vk::Result::eSuccess:
                break;
            case vk::Result::eErrorOutOfDateKHR:
                recreateSwapChain();
                break;
            case vk::Result::eSuboptimalKHR:
                spdlog::warn("suboptimal KHR");
                break;
            default:
                throw std::runtime_error("Failed to acquire swap chain image!");
        }

        if (images_in_flight[image_index] != vk::Fence{}) {
            while (vk::Result::eTimeout ==
                   logical_device.waitForFences(1, &images_in_flight[image_index], VK_TRUE, UINT64_MAX));
        }
        images_in_flight[image_index] = in_flight_fences[current_frame];

        vk::Semaphore wait_semaphores[1] = {image_available_semaphores[current_frame]};
        vk::PipelineStageFlags wait_stages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::Semaphore signal_semaphores[1] = {render_finished_semaphores[current_frame]};
        vk::SubmitInfo submit_info{
                .sType = vk::SubmitInfo::structureType,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = wait_semaphores,
                .pWaitDstStageMask = &wait_stages,
                .commandBufferCount = 1,
                .pCommandBuffers = &command_buffers[image_index],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = signal_semaphores,
        };

        while (vk::Result::eTimeout == logical_device.resetFences(1, &in_flight_fences[current_frame]));
        graphics_queue.submit(submit_info, in_flight_fences[current_frame]);

        vk::SwapchainKHR swap_chains[1] = {swap_chain};
        vk::PresentInfoKHR present_info{
                .sType = vk::PresentInfoKHR::structureType,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = signal_semaphores,
                .swapchainCount = 1,
                .pSwapchains = swap_chains,
                .pImageIndices = &image_index,
                .pResults = nullptr,
        };
        try {
            result = present_queue.presentKHR(present_info);
        } catch (vk::SystemError& err) {
            if (err.code() != vk::make_error_code(vk::Result::eErrorOutOfDateKHR)) {
                throw err;
            }
        }
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || frame_buffer_resized) {
            frame_buffer_resized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Some error happened in presentKHR");
        }
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }


    void recreateSwapChain() {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        logical_device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }


};

int main(int /*argc*/, char** /*argv*/) {
    HelloTriangleApplication app;
    try {
        app.run();
    } catch (const std::exception& e) {
        spdlog::log(spdlog::level::err, "{}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}