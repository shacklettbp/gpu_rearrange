#include "mgr.hpp"

#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#include <madrona/utils.hpp>

#include "sim.hpp"

using namespace madrona;
using namespace madrona::py;

namespace GPURearrange {

struct EpisodeData {
    void *episodeBuffer;
};

struct Manager::Impl {
    Config cfg;
    EpisodeData episodes;
    TrainingExecutor mwGPU;

    static inline Impl * init(const Config &cfg,
                              Span<Episode> episodes);
};

static EpisodeData setupEpisodes(Span<Episode> cpu_episodes)
{
    uint64_t total_bytes = sizeof(EpisodeManager);
    uint64_t episodes_offset 
        = utils::roundUp(total_bytes, alignof(Episode));
    total_bytes = episodes_offset + sizeof(Episode) * cpu_episodes.size();

    void *gpu_data = cu::allocGPU(total_bytes);
    void *episode_gpu_data = (char *)gpu_data + episodes_offset;

    EpisodeManager mgr_tmp {
        (Episode *)episode_gpu_data,
        uint32_t(cpu_episodes.size()),
        0,
    };

    cudaMemcpy(gpu_data, &mgr_tmp, sizeof(EpisodeManager),
               cudaMemcpyHostToDevice);

    cudaMemcpy(episode_gpu_data, cpu_episodes.data(),
               sizeof(Episode) * cpu_episodes.size(),
               cudaMemcpyHostToDevice);

    return EpisodeData {
        gpu_data,
    };
}

Manager::Impl * Manager::Impl::init(const Config &cfg,
                                    Span<Episode> episodes)
{
    EpisodeData gpu_episode_data = setupEpisodes(episodes);

    HeapArray<WorldInit> world_inits(cfg.numWorlds);

    for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
        world_inits[i] = WorldInit {
            (EpisodeManager *)gpu_episode_data.episodeBuffer,
        };
    }

    TrainingExecutor mwgpu_exec({
        .worldInitPtr = world_inits.data(),
        .numWorldInitBytes = sizeof(WorldInit),
        .numWorldDataBytes = sizeof(Sim),
        .worldDataAlignment = alignof(Sim),
        .numWorlds = cfg.numWorlds,
        .numExportedBuffers = 2,
        .gpuID = (uint32_t)cfg.gpuID,
        .renderWidth = cfg.renderWidth,
        .renderHeight = cfg.renderHeight,
    }, {
        "unused",
        { GPU_REARRANGE_SRC_LIST },
        { GPU_REARRANGE_COMPILE_FLAGS },
        CompileConfig::OptMode::Debug,
        CompileConfig::Executor::TaskGraph,
    });

    return new Impl {
        cfg,
        gpu_episode_data,
        std::move(mwgpu_exec),
    };
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg, {}))
{}

Manager::~Manager() {}

void Manager::step()
{
    impl_->mwGPU.run();
}

GPUTensor Manager::resetTensor() const
{
    void *dev_ptr = impl_->mwGPU.getExported(0);

    return GPUTensor(dev_ptr, GPUTensor::ElementType::Int32,
                     {impl_->cfg.numWorlds, 1}, impl_->cfg.gpuID);
}

GPUTensor Manager::moveActionTensor() const
{
    void *dev_ptr = impl_->mwGPU.getExported(1);

    return GPUTensor(dev_ptr, GPUTensor::ElementType::Int32,
                     {impl_->cfg.numWorlds, 1}, impl_->cfg.gpuID);
}

GPUTensor Manager::gpsCompassTensor() const
{
    void *dev_ptr = impl_->mwGPU.getExported(2);
    
    return GPUTensor(dev_ptr, GPUTensor::ElementType::Float32,
                     {impl_->cfg.numWorlds, 2, 2}, impl_->cfg.gpuID);
}

GPUTensor Manager::depthTensor() const
{
    void *dev_ptr = impl_->mwGPU.depthObservations();

    return GPUTensor(dev_ptr, GPUTensor::ElementType::Float32,
                     {impl_->cfg.numWorlds, impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 1}, impl_->cfg.gpuID);
}

}
