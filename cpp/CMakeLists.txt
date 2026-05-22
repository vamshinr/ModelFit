cmake_minimum_required(VERSION 3.16)
project(modelfit CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if user didn't specify.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Core library (CLI + GUI both depend on this)
# ---------------------------------------------------------------------------
set(CORE_SOURCES
    src/types.cpp
    src/quantization.cpp
    src/scoring.cpp
)

# Pick the right hardware backend per platform.
if(APPLE)
    list(APPEND CORE_SOURCES src/hardware_macos.cpp)
elseif(WIN32)
    list(APPEND CORE_SOURCES src/hardware_windows.cpp)
else()
    list(APPEND CORE_SOURCES src/hardware_linux.cpp)
endif()

add_library(modelfit_core STATIC ${CORE_SOURCES})
target_include_directories(modelfit_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Platform-specific link libraries
if(WIN32)
    target_link_libraries(modelfit_core PUBLIC dxgi)
endif()

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
add_executable(modelfit-cli src/cli_main.cpp)
target_link_libraries(modelfit-cli PRIVATE modelfit_core)
if(WIN32)
    set_target_properties(modelfit-cli PROPERTIES SUFFIX ".exe")
endif()

install(TARGETS modelfit-cli RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Optional GUI — needs Dear ImGui + GLFW + OpenGL
# Enable with -DMODELFIT_GUI=ON. Fetches deps from GitHub on first config.
# ---------------------------------------------------------------------------
option(MODELFIT_GUI "Build the desktop GUI (requires network for first config)" OFF)

if(MODELFIT_GUI)
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.5
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(glfw imgui)

    # Dear ImGui doesn't ship its own CMakeLists, so we compile the core +
    # the GLFW + OpenGL3 backends ourselves.
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_link_libraries(imgui PUBLIC glfw)

    find_package(OpenGL REQUIRED)
    add_executable(modelfit-gui
        src/gui_main.cpp
        src/glossary_data.cpp
        src/glossary_search.cpp)
    target_link_libraries(modelfit-gui PRIVATE modelfit_core imgui OpenGL::GL)
    if(APPLE)
        target_link_libraries(modelfit-gui PRIVATE
            "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
    elseif(UNIX)
        target_link_libraries(modelfit-gui PRIVATE dl pthread X11)
    endif()
    install(TARGETS modelfit-gui RUNTIME DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# Regenerate the catalog header from the Python catalog.
# Optional convenience target: `cmake --build . --target gen_catalog`
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(gen_catalog
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/gen_catalog.py
                > ${CMAKE_SOURCE_DIR}/include/modelfit/catalog_data.h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
        COMMENT "Regenerating cpp/include/modelfit/catalog_data.h"
        VERBATIM
    )
endif()
