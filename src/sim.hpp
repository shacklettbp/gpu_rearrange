#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/render.hpp>

#include "init.hpp"

namespace GPURearrange {

using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;

class Engine;

struct WorldReset {
    int32_t resetNow;
};

struct DynamicObject : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    ObjectID,
    madrona::phys::CollisionAABB,
    madrona::phys::broadphase::LeafID
> {};

struct Action {
    int32_t action;
};

struct Goal {
    Position objectStartingPosition;
    Position goalPosition;
    madrona::Entity goalEntity;
};

struct GPSCompassObs {
    madrona::math::Vector2 toObjectStartPolar;
    madrona::math::Vector2 toGoalPolar;
};

struct Reward {
    float reward;
};

struct ObjectHeld {
    int32_t held;
};

static_assert(sizeof(Action) == sizeof(int32_t));
static_assert(sizeof(ObjectHeld) == sizeof(int32_t));

static_assert(sizeof(GPSCompassObs) == sizeof(float) * 4);
static_assert(sizeof(Reward) == sizeof(float));

struct Agent : public madrona::Archetype<
    Position,
    Rotation,
    madrona::phys::CollisionAABB,
    madrona::phys::broadphase::LeafID,
    madrona::render::ActiveView,
    Action,
    ObjectHeld,
    Goal,
    GPSCompassObs,
    Reward
> {};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry);

    static void setupTasks(madrona::TaskGraph::Builder &builder);

    Sim(Engine &ctx, const WorldInit &init);

    EpisodeManager *episodeMgr;

    madrona::Entity agent;
    madrona::Entity staticEntity;
    madrona::Entity *dynObjects;
    madrona::CountT numObjects;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
