cmake_minimum_required(VERSION 3.22)

# Honor MSVC_DEBUG_INFORMATION_FORMAT so the rg target can pick /ZI (Edit & Continue).
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()

project(webgpu_app CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(SHARED_SOURCES
    src/Framework.cpp
    src/Renderer.cpp
    src/ClusteredMesh.cpp
)

# ---------------------------------------------------------------------------
# Emscripten (WebAssembly / WebGPU via emdawnwebgpu port)
# ---------------------------------------------------------------------------
if(EMSCRIPTEN)
    FetchContent_Declare(
        emdawnwebgpu
        URL      https://github.com/google/dawn/releases/download/v20260420.230544/emdawnwebgpu_pkg-v20260420.230544.zip
        URL_HASH SHA256=e7be04cc77acdc608dd80e0eafefeda92c5acc9fe45133ee91cef7a400250c8c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(emdawnwebgpu)

    set(APP_MAIN src/main_web.cpp)

# ---------------------------------------------------------------------------
# Native Windows — prebuilt Dawn binaries from Dawn v20260420.230544
# ---------------------------------------------------------------------------
elseif(WIN32)
    set(DAWN_HASH 541d4aa11ac0768fa029db8e6cc2d406a49a9c99f485eccb42455c046e3c292b)
    FetchContent_Declare(
        dawn_native
        URL      https://github.com/google/dawn/releases/download/v20260420.230544/Dawn-78a1bb0b179f2eaeb9e881f6d4bae4d545dc6518-windows-latest-Release.tar.gz
        URL_HASH SHA256=${DAWN_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(SDL3_VER "3.4.4")
    FetchContent_Declare(
        sdl3
        URL      https://github.com/libsdl-org/SDL/releases/download/release-${SDL3_VER}/SDL3-devel-${SDL3_VER}-VC.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    FetchContent_MakeAvailable(dawn_native sdl3)

    # Point find_package to the extracted CMake config directories.
    set(Dawn_DIR "${dawn_native_SOURCE_DIR}/lib/cmake/Dawn")

    file(GLOB_RECURSE _sdl3_config "${sdl3_SOURCE_DIR}/SDL3Config.cmake")
    if(NOT _sdl3_config)
        message(FATAL_ERROR "SDL3Config.cmake not found under ${sdl3_SOURCE_DIR}")
    endif()
    list(GET _sdl3_config 0 _sdl3_config)
    get_filename_component(SDL3_DIR "${_sdl3_config}" DIRECTORY)

    find_package(Threads REQUIRED)
    find_package(Dawn  REQUIRED)
    find_package(SDL3  REQUIRED CONFIG)

    set(APP_MAIN src/main_SDL.cpp)

else()
    message(FATAL_ERROR "Only Emscripten and Windows are supported by this project.")
endif()

# ---------------------------------------------------------------------------
# Shared app target
# ---------------------------------------------------------------------------
add_executable(app ${APP_MAIN} ${SHARED_SOURCES})

if(EMSCRIPTEN)
    set(DAWN_PORT "${emdawnwebgpu_SOURCE_DIR}/emdawnwebgpu.port.py")

    target_compile_options(app PRIVATE "--use-port=${DAWN_PORT}" "-Oz")
    target_link_options(app PRIVATE
        "--use-port=${DAWN_PORT}"
        "-sASYNCIFY=0"
        "-sALLOW_MEMORY_GROWTH=1"
        "--shell-file=${CMAKE_SOURCE_DIR}/src/shell.html"
        "-Oz"
        "--preload-file=${CMAKE_SOURCE_DIR}/assets@/assets"
        "-sINVOKE_RUN=0"
    )

    set_target_properties(app PROPERTIES SUFFIX ".html")

    add_custom_command(TARGET app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
            "$<TARGET_FILE_DIR:app>/app.html"
            "$<TARGET_FILE_DIR:app>/index.html"
        COMMENT "Renaming app.html to index.html for GitHub Pages"
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(app PRIVATE "-flto")
        target_link_options(app PRIVATE
            "--closure=1"
            "-flto"
            "-sWASM=2"
            "-sENVIRONMENT=web"
            "-sMALLOC=emmalloc"
            "-sASSERTIONS=0"
            "-sSAFE_HEAP=0"
            "-sSTACK_OVERFLOW_CHECK=0"
            "-sDEMANGLE_SUPPORT=0"
            "-sGL_ASSERTIONS=0"
        )
    endif()

elseif(WIN32)
    target_link_libraries(app PRIVATE dawn::webgpu_dawn SDL3::SDL3)

    add_executable(nanite_builder src/mainNaniteBuilder.cpp src/ClusteredMesh.cpp)
    target_link_libraries(nanite_builder PRIVATE dawn::webgpu_dawn)

    # Dear ImGui (vendored): core + SDL3 platform + WebGPU(Dawn) renderer backends, each its own
    # TU -- the same set build_rg.bat compiles. The Dawn backend needs IMGUI_IMPL_WEBGPU_BACKEND_DAWN.
    set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/src/extern/imgui)
    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
        ${IMGUI_DIR}/backends/imgui_impl_wgpu.cpp
    )

    # RenderGraph standalone smoke test. RenderGraph_main.cpp is #included at the bottom of
    # RenderGraph.cpp, so that single TU is the whole driver; the ImGui TUs link alongside it
    # (mirrors build_rg.bat). Built for Edit & Continue: Debug gets /ZI + incremental link, /MD to
    # match the prebuilt release Dawn/SDL3. Set rg as the startup project and run Debug to use it.
    add_executable(rg 
        src/RenderGraph.cpp
        src/RenderGraph_main.cpp
        src/RenderGraph_deferred.cpp
        src/RenderGraph_pathtracer.cpp
        src/RenderGraph_particles.cpp
        src/RenderGraph_bake.cpp
        src/RenderGraph.h
        src/imgui_layer.h ${IMGUI_SOURCES})
    # RenderGraph_main.cpp is #included at the bottom of RenderGraph.cpp; list it (and the header)
    # so they show in Solution Explorer, but HEADER_FILE_ONLY keeps it out of the build so it isn't
    # compiled as a second TU.
    set_source_files_properties(
        src/RenderGraph_main.cpp
        src/RenderGraph_deferred.cpp
        src/RenderGraph_pathtracer.cpp
        src/RenderGraph_particles.cpp
        src/RenderGraph_bake.cpp
        PROPERTIES HEADER_FILE_ONLY TRUE)
    target_link_libraries(rg PRIVATE dawn::webgpu_dawn SDL3::SDL3 dxguid onecore)
    target_include_directories(rg PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
    target_compile_definitions(rg PRIVATE IMGUI_IMPL_WEBGPU_BACKEND_DAWN)
    set_target_properties(rg PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
        MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug>:EditAndContinue>"
        VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:rg>"
    )

    # Copy SDL3.dll, dxil.dll, and dxcompiler.dll next to each exe at runtime.
    set(DXIL_DIR "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/Redist/D3D/x64")
    file(GLOB_RECURSE _sdl3_dll "${sdl3_SOURCE_DIR}/SDL3.dll")
    list(FILTER _sdl3_dll INCLUDE REGEX "x64")
    if(NOT _sdl3_dll)
        message(FATAL_ERROR "SDL3.dll (x64) not found under ${sdl3_SOURCE_DIR}")
    endif()
    list(GET _sdl3_dll 0 _sdl3_dll)
    foreach(TGT app rg)
        foreach(DLL dxil.dll dxcompiler.dll d3dcompiler_47.dll)
            add_custom_command(TARGET ${TGT} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DXIL_DIR}/${DLL}"
                    "$<TARGET_FILE_DIR:${TGT}>/${DLL}"
                COMMENT "Copying ${DLL}"
            )
        endforeach()
        add_custom_command(TARGET ${TGT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_sdl3_dll}"
                "$<TARGET_FILE_DIR:${TGT}>/SDL3.dll"
            COMMENT "Copying SDL3.dll"
        )
    endforeach()
endif()
