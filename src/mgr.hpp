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
        const char *episodeFile;
        const char *dataDir;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::GPUTensor resetTensor() const;
    MADRONA_IMPORT madrona::py::GPUTensor moveActionTensor() const;
    MADRONA_IMPORT madrona::py::GPUTensor gpsCompassTensor() const;
    MADRONA_IMPORT madrona::py::GPUTensor depthTensor() const;
    MADRONA_IMPORT madrona::py::GPUTensor rgbTensor() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}
