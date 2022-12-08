set(SIMULATOR_SRCS
    sim.hpp sim.cpp
)

madrona_build_compile_defns(
    OUT_TARGET
        gpu_rearrange_gpu_srcs
    SOURCES_DEFN
        GPU_REARRANGE_SRC_LIST
    FLAGS_DEFN
        GPU_REARRANGE_COMPILE_FLAGS 
    SRCS
        ${SIMULATOR_SRCS}
)

add_library(gpu_rearrange_mgr SHARED
    mgr.hpp mgr.cpp
    data.hpp data.cpp
    gltf.hpp gltf.cpp
)

target_link_libraries(gpu_rearrange_mgr PRIVATE
    madrona_python_utils
    madrona_mw_gpu
    madrona_common
    gpu_rearrange_gpu_srcs
    simdjson
    ZLIB::ZLIB
)

madrona_python_module(gpu_rearrange_python
    bindings.cpp
)

target_link_libraries(gpu_rearrange_python PRIVATE
    madrona_hdrs
    gpu_rearrange_mgr
)
