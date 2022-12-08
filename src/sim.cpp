#include "sim.hpp"

#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::phys;
using namespace madrona::math;

namespace GPURearrange {

static constexpr inline CountT max_instances = 45;


void Sim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<ObjectHeld>();
    registry.registerComponent<Goal>();

    registry.registerComponent<GPSCompassObs>();
    registry.registerComponent<Reward>();

    registry.registerSingleton<WorldReset>();
    registry.exportSingleton<WorldReset>(0);

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<Agent>();

    registry.exportColumn<Agent, Action>(1);
    registry.exportColumn<Agent, GPSCompassObs>(2);
}

static void resetWorld(Engine &ctx)
{
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx = 
        episode_mgr.episodeOffset.fetch_add(1, std::memory_order_relaxed);

    Episode episode = episode_mgr.episodes[episode_idx];

    assert(episode.numInstances <= max_instances);

    Entity *dyn_entities = ctx.data().dynObjects;
    for (CountT i = 0; i < CountT(episode.numInstances); i++) {
        InstanceInit instance_init =
            episode_mgr.instanceInits[episode.instanceOffset + i];

        Entity dyn_entity = dyn_entities[i];

        ctx.getUnsafe<render::ObjectID>(dyn_entity).idx =
            instance_init.objectIndex;
        ctx.getUnsafe<Position>(dyn_entity) = instance_init.pos;
        ctx.getUnsafe<Rotation>(dyn_entity) = instance_init.rot;
    }

    Entity agent_entity = ctx.data().agent;
    ctx.getUnsafe<Position>(agent_entity) = episode.agentPos;
    ctx.getUnsafe<Rotation>(agent_entity) = episode.agentRot;

    Goal &goal_data = ctx.getUnsafe<Goal>(agent_entity);
    goal_data.objectStartingPosition = episode_mgr.instanceInits[
        episode.instanceOffset + episode.targetIdx].pos;
    goal_data.goalPosition = episode.goalPos;
    goal_data.goalEntity = dyn_entities[episode.targetIdx];

    ctx.getSingleton<broadphase::BVH>().rebuild();
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (!reset.resetNow) {
        return;
    }
    reset.resetNow = false;

    resetWorld(ctx);
}

inline void actionSystem(Engine &, const Action &action,
                         Position &pos, Rotation &rot)
{
    constexpr float turn_angle = helpers::toRadians(10.f);

    switch(action.action) {
    case 0: {
        // Implement stop
    } break;
    case 1: {
        Vector3 fwd = rot.rotateDir({0, 0, -1});
        pos += fwd;
    } break;
    case 2: {
        const Quat left_rot = Quat::angleAxis(turn_angle, {0, 1, 0});
        rot = rot * left_rot;
    } break;
    case 3: {
        const Quat right_rot = Quat::angleAxis(-turn_angle, {0, 1, 0});
        rot = rot * right_rot;
    } break;
    default:
        break;
    }
}

inline void learningOutputsSystem(Engine &ctx,
                                  GPSCompassObs &obs,
                                  Reward &reward,
                                  const Position agent_pos,
                                  const Rotation agent_rot,
                                  const Goal goal)
{
    Vector3 to_object_start =
        agent_rot.rotateDir(goal.objectStartingPosition - agent_pos);
    Vector3 to_goal = 
        agent_rot.rotateDir(goal.goalPosition - agent_pos);
    
    auto cartesianToPolar = [](float x, float y) {
        float rho = Vector2 {x, y}.length();
        float phi = atan2f(y, x);

        return Vector2 {rho, phi};
    };

    obs.toObjectStartPolar =
        cartesianToPolar(to_object_start.z, to_object_start.x);
    obs.toGoalPolar = cartesianToPolar(to_goal.z, to_goal.x);

    Position cur_object_pos = ctx.getUnsafe<Position>(goal.goalEntity);
    
    Vector3 to_object_cur = cur_object_pos - agent_pos;

    // Terrible fake reward
    reward.reward = 1.f / to_object_cur.length();
}

void Sim::setupTasks(TaskGraph::Builder &builder)
{
    auto reset_sys =
        builder.parallelForNode<Engine, resetSystem, WorldReset>({});

    auto action_sys = builder.parallelForNode<Engine, actionSystem,
        Action, Position, Rotation>({reset_sys});

    auto phys_sys = RigidBodyPhysicsSystem::setupTasks(builder, {action_sys});

    auto sim_done = phys_sys;

    auto phys_cleanup_sys = RigidBodyPhysicsSystem::setupCleanupTasks(builder,
        {sim_done});

    auto learning_sys = builder.parallelForNode<Engine, learningOutputsSystem,
        GPSCompassObs, Reward, Position, Rotation, Goal>({sim_done});

    auto renderer_sys = render::RenderingSystem::setupTasks(builder,
        {sim_done});

    (void)phys_cleanup_sys;
    (void)learning_sys;
    (void)renderer_sys;

    printf("Setup done\n");
}


Sim::Sim(Engine &ctx, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    RigidBodyPhysicsSystem::init(ctx, 45, 100 * 50);

    render::RenderingSystem::init(ctx);

    broadphase::BVH &bp_bvh = ctx.getSingleton<broadphase::BVH>();
    
    agent = ctx.makeEntityNow<Agent>();
    ctx.getUnsafe<render::ActiveView>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, {0, 1, 0});
    ctx.getUnsafe<broadphase::LeafID>(agent) =
        bp_bvh.reserveLeaf();

    dynObjects = (Entity *)malloc(sizeof(Entity) * (max_instances - 1));

    for (CountT i = 0; i < max_instances - 1; i++) {
        dynObjects[i] = ctx.makeEntityNow<DynamicObject>();
        ctx.getUnsafe<broadphase::LeafID>(dynObjects[i]) =
            bp_bvh.reserveLeaf();
    }

    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
