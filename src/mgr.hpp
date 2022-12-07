#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace GPURearrange {

class Manager {
public:
    struct Config {
        int gpuID;
        uint32_t numWorlds;
        uint32_t renderWidth;
        uint32_t renderHeight;
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    madrona::py::GPUTensor resetTensor() const;
    madrona::py::GPUTensor moveActionTensor() const;
    madrona::py::GPUTensor gpsCompassTensor() const;
    madrona::py::GPUTensor depthTensor() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}
