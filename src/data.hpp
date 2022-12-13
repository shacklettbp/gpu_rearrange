#pragma once

#include <madrona/importer.hpp>

#include <vector>

#include "init.hpp"

namespace GPURearrange {

struct TrainingData {
    std::vector<Episode> episodes;
    std::vector<InstanceInit> instances;
    std::vector<std::vector<madrona::math::Vector3>> positions;
    std::vector<std::vector<madrona::math::Vector3>> normals;
    std::vector<std::vector<madrona::math::Vector2>> uvs;
    std::vector<std::vector<uint32_t>> indices;
    std::vector<std::vector<madrona::imp::SourceMesh>> meshes;
    std::vector<madrona::imp::SourceObject> objects;

    static TrainingData load(const char *episode_file, const char *data_dir);
};

}
