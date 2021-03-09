git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat --disable-metrics
.\vcpkg\vcpkg.exe --triplet x64-windows-static install fmt spdlog glfw3