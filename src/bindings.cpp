#include "mgr.hpp"

#include <madrona/macros.hpp>

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
    nb::class_<Manager>(m, "RearrangeSimulator")
        .def(nb::init<Manager::Config>())
        .def("gps_compass_tensor", &Manager::gpsCompassTensor)
        .def("depth_tensor", &Manager::depthTensor)
    ;
}

}
