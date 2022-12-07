#pragma once 

#include <atomic>
#include <cstdint>

#include <madrona/components.hpp>

namespace GPURearrange {

struct InstanceInit {
    int32_t objectIndex;
    madrona::base::Position pos;
    madrona::base::Rotation rot;
};

struct Episode {
    int32_t instanceOffset;
    int32_t numInstances;

    int32_t targetIdx;
    madrona::base::Position agentPos;
    madrona::base::Rotation agentRot;

    madrona::base::Position goalPos;
};

struct EpisodeManager {
    InstanceInit *instanceInits;
    Episode *episodes;

    uint32_t numEpisodes;
    std::atomic_uint32_t episodeOffset;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
};

}
