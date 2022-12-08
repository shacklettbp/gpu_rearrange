#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/heap_array.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace GPURearrange {

NB_MODULE(gpu_rearrange_python, m) {
    nb::class_<Manager> (m, "RearrangeSimulator")
        .def("__init__", [](Manager *self, int64_t gpu_id,
                            int64_t num_worlds, int64_t render_width,
                            int64_t render_height, const char *episode_file,
                            const char *data_dir) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .renderWidth = (uint32_t)render_width,
                .renderHeight = (uint32_t)render_height,
                .episodeFile = episode_file,
                .dataDir = data_dir,
            });
        }, nb::arg("gpu_id"), nb::arg("num_worlds"), nb::arg("render_width"),
           nb::arg("render_height"), nb::arg("episode_file"),
           nb::arg("data_dir"))
        .def("step", &Manager::step)
        .def("gps_compass_tensor", &Manager::gpsCompassTensor)
        .def("depth_tensor", &Manager::depthTensor)
    ;
}

}
