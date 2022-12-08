#pragma once

#include <madrona/scene.hpp>

#include <vector>

#include "init.hpp"

namespace GPURearrange {

struct TrainingData {
    std::vector<Episode> episodes;
    std::vector<InstanceInit> instances;
    std::vector<std::vector<madrona::render::SourceVertex>> vertices;
    std::vector<std::vector<uint32_t>> indices;
    std::vector<std::vector<madrona::render::SourceMesh>> meshes;
    std::vector<madrona::render::SourceObject> objects;

    static TrainingData load(const char *episode_file, const char *data_dir);
};

}
