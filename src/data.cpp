#include "data.hpp"
#include "gltf.hpp"

#include <madrona/macros.hpp>

#include <simdjson.h>
#include <zlib.h>

#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <vector>

using namespace madrona;

namespace GPURearrange {

static inline void checkSIMDJsonResult(
    simdjson::error_code err,
    const char *file,
    int line,
    const char *func_name)
{
    if (err) {
        fatal(file, line, func_name, "Failed to parse JSON: %s",
              simdjson::error_message(err));
    }
}

#define REQ_JSON(err) checkSIMDJsonResult(err, __FILE__, \
    __LINE__, MADRONA_COMPILER_FUNCTION_NAME);

template <typename T>
static inline T getArrayElem(
    simdjson::simdjson_result<simdjson::ondemand::array_iterator> iter)
{
    auto r = *iter;

    T v;
    REQ_JSON(r.get(v));

    return v;
}

static inline simdjson::simdjson_result<simdjson::ondemand::array_iterator>
getArrayBegin(simdjson::simdjson_result<simdjson::ondemand::value> v)
{
    simdjson::ondemand::array arr;
    REQ_JSON(v.get(arr));

    return arr.begin();
}

static inline math::Quat getQuat(
    simdjson::simdjson_result<simdjson::ondemand::value> value,
    bool w_first)
{
    math::Quat q;

    auto iter = getArrayBegin(value);

    if (w_first) {
        q.w = float(getArrayElem<double>(iter));
        ++iter;
    }

    q.x = float(getArrayElem<double>(iter));
    ++iter;

    q.y = float(getArrayElem<double>(iter));
    ++iter;

    q.z = float(getArrayElem<double>(iter));

    if (!w_first) {
        ++iter;
        q.w = float(getArrayElem<double>(iter));
    }

    return q;
}

static inline math::Vector3 getVec3(
    simdjson::simdjson_result<simdjson::ondemand::value> value)
{
    math::Vector3 v;

    auto iter = getArrayBegin(value);

    v.x = float(getArrayElem<double>(iter));
    v.y = float(getArrayElem<double>(++iter));
    v.z = float(getArrayElem<double>(++iter));

    return v;
}

static inline math::Vector4 getVec4(
    simdjson::simdjson_result<simdjson::ondemand::value> value)
{
    math::Vector4 v;

    auto iter = getArrayBegin(value);

    v.x = float(getArrayElem<double>(iter));
    v.y = float(getArrayElem<double>(++iter));
    v.z = float(getArrayElem<double>(++iter));
    v.w = float(getArrayElem<double>(++iter));

    return v;
}

struct ParsedInstance {
    int64_t objID;
    math::Vector3 translation;
    math::Quat rotation;
    bool dynamic;
};

struct ParsedScene {
    int64_t stageObjID;
    math::Vector3 stageTranslation;
    math::Quat stageRotation;
    std::vector<ParsedInstance> staticInstances;
};

struct ParsedEpisode {
    const ParsedScene *scene;
    std::string_view objDir;
    math::Vector3 agentPos;
    math::Quat agentRot;
    std::vector<ParsedInstance> dynamicInstances;
};

struct ParseData {
    const char *dataDir;
    std::unordered_map<std::string, int64_t> parsedObjs;
    std::unordered_map<std::string, ParsedScene> parsedScenes;
    simdjson::ondemand::parser sceneParser;

    TrainingData trainData {};
};

static int64_t parseObject(std::string_view obj_path,
                           ParseData &parse_data,
                           math::Vector3 right = {1, 0, 0},
                           math::Vector3 up = {0, 1, 0},
                           math::Vector3 fwd = {0, 0, 1})
{
    using namespace std;

    auto parsed_stages_iter =
        parse_data.parsedObjs.find(std::string(obj_path));
    if (parsed_stages_iter != parse_data.parsedObjs.end()) {
        return parsed_stages_iter->second;
    }

    std::cout << "'" << obj_path << "' " << std::endl;

    int64_t obj_id = loadAndParseGLTF(obj_path, right, up, fwd,
                                      parse_data.trainData);

    auto [iter, success] = parse_data.parsedObjs.emplace(obj_path, obj_id);
    assert(success);

    return obj_id;
}

static const ParsedScene * parseScene(std::string_view scene_path,
                                      ParseData &parse_data)
{
    using namespace std;

    auto parsed_scenes_iter = parse_data.parsedScenes.find(std::string(scene_path));
    if (parsed_scenes_iter != parse_data.parsedScenes.end()) {
        return &parsed_scenes_iter->second;
    }

    auto sub_path = scene_path.substr(scene_path.find('/'));

    string full_path = parse_data.dataDir;
    full_path += sub_path;

    simdjson::padded_string json_data;
    REQ_JSON(simdjson::padded_string::load(full_path).get(json_data));

    simdjson::ondemand::document scene_json;
    REQ_JSON(parse_data.sceneParser.iterate(json_data).get(scene_json));

    string_view stage_name;
    auto stage_inst = scene_json["stage_instance"];
    REQ_JSON(stage_inst["template_name"].get(stage_name));

    auto stage_translation = getVec3(stage_inst["translation"]);
    auto stage_rotation = getQuat(stage_inst["rotation"], true);

    // FIXME
    std::string stage_path = parse_data.dataDir;
    stage_path += "replica_cad/stages/";
    stage_path += stage_name;
    stage_path += ".glb";

    int64_t stage_obj_id = parseObject(stage_path, parse_data);

    vector<ParsedInstance> static_instances;

    simdjson::ondemand::array instances_arr;
    REQ_JSON(scene_json["object_instances"].get(instances_arr));
    for (auto instance : instances_arr) {
        string_view motion_type;
        REQ_JSON(instance["motion_type"].get(motion_type));
        if (motion_type != "STATIC"sv) {
            continue;
        }

        string_view obj_name;
        REQ_JSON(instance["template_name"].get(obj_name));

        std::string obj_path = parse_data.dataDir;
        obj_path += "replica_cad/objects/";
        obj_path += obj_name;
        obj_path += ".glb";

        int64_t inst_obj_id = parseObject(obj_path, parse_data);

        math::Vector3 translation = getVec3(instance["translation"]);
        math::Quat rotation = getQuat(instance["rotation"], true);

        static_instances.push_back({
            .objID = inst_obj_id,
            .translation = translation,
            .rotation = rotation,
            .dynamic = false,
        });
    }

    ParsedScene parsed_scene {
        .stageObjID = stage_obj_id,
        .stageTranslation = stage_translation,
        .stageRotation = stage_rotation,
        .staticInstances = std::move(static_instances),
    };

    auto [iter, success] =
        parse_data.parsedScenes.emplace(scene_path, std::move(parsed_scene));
    assert(success);

    return &iter->second;
}

static inline math::Quat rotToQuat(
    math::Vector3 a, math::Vector3 b, math::Vector3 c)
{
    /* Modified from glm::quat_cast
 ==============================================================================
The MIT License
-------------------------------------------------------------------------------
Copyright (c) 2005 - G-Truc Creation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

    float fourXSquaredMinus1 = a.x - b.y - c.z;
	float fourYSquaredMinus1 = b.y - a.x - c.z;
	float fourZSquaredMinus1 = c.z - a.x - b.y;
	float fourWSquaredMinus1 = a.x + b.y + c.z;

	int biggestIndex = 0;
	float fourBiggestSquaredMinus1 = fourWSquaredMinus1;
	if(fourXSquaredMinus1 > fourBiggestSquaredMinus1)
	{
		fourBiggestSquaredMinus1 = fourXSquaredMinus1;
		biggestIndex = 1;
	}
	if(fourYSquaredMinus1 > fourBiggestSquaredMinus1)
	{
		fourBiggestSquaredMinus1 = fourYSquaredMinus1;
		biggestIndex = 2;
	}
	if(fourZSquaredMinus1 > fourBiggestSquaredMinus1)
	{
		fourBiggestSquaredMinus1 = fourZSquaredMinus1;
		biggestIndex = 3;
	}

	float biggestVal = sqrtf(fourBiggestSquaredMinus1 + 1.f) * 0.5f;
	float mult = 0.25f / biggestVal;

	switch(biggestIndex) {
	case 0:
		return {
            biggestVal, 
            (b.z - c.y) * mult,
            (c.x - a.z) * mult,
            (a.y - b.x) * mult,
        };
	case 1:
		return {
            (b.z - c.y) * mult,
            biggestVal,
            (a.y + b.x) * mult,
            (c.x + a.z) * mult,
        }; case 2:
		return {
            (c.x - a.z) * mult,
            (a.y + b.x) * mult,
            biggestVal,
            (b.z + c.y) * mult,
        };
	case 3:
		return {
            (a.y - b.x) * mult,
            (c.x + a.z) * mult,
            (b.z + c.y) * mult,
            biggestVal,
        };
	default:
		assert(false);
	}
}

TrainingData TrainingData::load(const char *episode_file,
                                const char *data_dir)
{
    using namespace std;

    size_t num_bytes = filesystem::file_size(episode_file);
    gzFile gz = gzopen(episode_file, "rb");
    if (gz == nullptr) {
        FATAL("Failed to open %s", episode_file);
    }

    vector<uint8_t> out_data {};

    size_t cur_out_size = num_bytes * 2;
    int cur_decompressed = 0;
    size_t total_decompressed = 0;
    for (; !gzeof(gz) && cur_decompressed >= 0;
         cur_decompressed = gzread(gz, out_data.data() + total_decompressed,
                                   cur_out_size - total_decompressed),
         total_decompressed += cur_decompressed, cur_out_size *= 2) {
        out_data.resize(cur_out_size + simdjson::SIMDJSON_PADDING);
    }

    if (cur_decompressed == -1) {
        FATAL("Failed to decompress %s", episode_file);
    }

    gzclose(gz);

    simdjson::ondemand::parser main_parser;
    simdjson::ondemand::document episode_json;
    REQ_JSON(main_parser.iterate(out_data.data(), total_decompressed,
                                 out_data.size()).get(episode_json));

    simdjson::ondemand::array episodes_array;
    REQ_JSON(episode_json["episodes"].get(episodes_array));

    ParseData parse_data;
    parse_data.dataDir = data_dir;

    for (auto episode : episodes_array) {
        string_view scene_path;
        REQ_JSON(episode["scene_id"].get(scene_path));

        const ParsedScene *scene =
            parseScene(scene_path, parse_data);

        math::Vector3 start_pos = getVec3(episode["start_position"]);
        math::Quat start_rot = getQuat(episode["start_rotation"], true);

        auto obj_config_dirs_iter =
            getArrayBegin(episode["additional_obj_config_paths"]);

        string_view obj_config_dir;
        REQ_JSON((*obj_config_dirs_iter).get(obj_config_dir));
        obj_config_dir = obj_config_dir.substr(obj_config_dir.find('/'));

        simdjson::ondemand::array rigid_objs_arr;
        REQ_JSON(episode["rigid_objs"].get(rigid_objs_arr));

        vector<ParsedInstance> dyn_insts;
        for (auto rigid_obj : rigid_objs_arr) {
            auto obj_arr_iter = getArrayBegin(rigid_obj);

            std::string_view obj_config_path;
            REQ_JSON((*obj_arr_iter).get(obj_config_path));

            auto txfm_iter = getArrayBegin(*++obj_arr_iter);

            auto row0 = getVec4(*txfm_iter);
            auto row1 = getVec4(*++txfm_iter);
            auto row2 = getVec4(*++txfm_iter);

            auto txfm = math::Mat3x4::fromRows(row0, row1, row2);
            auto translation = txfm.cols[3];
            math::Vector3 scale {
                txfm.cols[0].length(),
                txfm.cols[1].length(),
                txfm.cols[2].length(),
            };

            if (dot(cross(txfm.cols[0], txfm.cols[1]), txfm.cols[2]) < 0.f) {
                scale.x *= -1.f;
            }

            assert(fabsf(scale.x - 1.f) < 0.1);
            assert(fabsf(scale.y - 1.f) < 0.1);
            assert(fabsf(scale.z - 1.f) < 0.1);

            auto v1 = (txfm.cols[0] / scale.x).normalize();
            auto v2 = txfm.cols[1] / scale.y;
            auto v3 = txfm.cols[2] / scale.z;

            v2 = (v2 - dot(v2, v1) * v1).normalize();
            v3 = (v3 - dot(v3, v1) * v1);
            v3 -= dot(v3, v2) * v2;
            v3 = v3.normalize();

            math::Quat rot = rotToQuat(v1, v2, v3);

            std::string obj_path = data_dir;
            obj_path += obj_config_dir;
            obj_path += "/../meshes/";
            auto obj_prefix = obj_config_path.substr(0, obj_config_path.find('.'));
            obj_path += obj_prefix;
            obj_path += "/google_16k/textured.glb";

            int64_t inst_obj_id = parseObject(obj_path, parse_data,
                                              {-1, 0, 0},
                                              {0, 1, 0},
                                              {0, 0, -1});

            dyn_insts.push_back(ParsedInstance {
                .objID = inst_obj_id,
                .translation = translation,
                .rotation = rot,
                .dynamic = true,
            });
        }

        int32_t inst_offset = parse_data.trainData.instances.size();
        parse_data.trainData.instances.push_back({
            int32_t(scene->stageObjID),
            scene->stageTranslation,
            scene->stageRotation,
        });

        for (const ParsedInstance &inst : scene->staticInstances) {
            parse_data.trainData.instances.push_back({
                int32_t(inst.objID),
                inst.translation,
                inst.rotation,
            });
        }

        int32_t dyn_offset = parse_data.trainData.instances.size();

        for (const ParsedInstance &inst : dyn_insts) {
            parse_data.trainData.instances.push_back({
                int32_t(inst.objID),
                inst.translation,
                inst.rotation,
            });

            // HACK
            if (parse_data.trainData.instances.size() - inst_offset == 45) {
                break;
            }
        }

        int32_t inst_end_offset = parse_data.trainData.instances.size();
        int32_t num_instances = inst_end_offset - inst_offset;
        assert(num_instances == 45); // HACK

        parse_data.trainData.episodes.push_back({
            inst_offset,
            num_instances,
            dyn_offset - inst_offset,
            start_pos,
            start_rot,
            dyn_insts[0].translation,
        });

        assert(num_instances < 100);
    }

    std::cout << parse_data.trainData.episodes.size() << std::endl;
    return std::move(parse_data.trainData);
}

}
