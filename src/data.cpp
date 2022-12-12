#include "data.hpp"
#include "gltf.hpp"

#include <madrona/macros.hpp>

#include <simdjson.h>
#include <zlib.h>
#include <libxml/xmlreader.h>

#include <charconv>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <vector>

using namespace madrona;

namespace GPURearrange {

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
    math::Vector3 scale;
    bool dynamic;
};

struct ParsedScene {
    int64_t stageObjID;
    math::Vector3 stageTranslation;
    math::Quat stageRotation;
    std::vector<ParsedInstance> additionalInstances;
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

struct URDFLinkState {
    math::Vector3 translation;
    math::Quat rotation;
    std::string renderModel;
};

struct URDFJointState {
    math::Vector3 translation;
    math::Quat rotation;
    std::string parent;
    std::string child;
};

struct URDFGraphNode {
    struct Edge {
        math::Mat3x4 transform;
        int32_t child;
    };

    std::string model;
    math::Mat3x4 transform;
    int32_t parent;
    std::vector<Edge> edges;
};

static math::Quat urdfReadRPY(xmlTextReaderPtr reader)
{
    auto xml_rpy =
        xmlTextReaderGetAttribute(reader, BAD_CAST "rpy");
    if (xml_rpy == nullptr) {
        return { 1, 0, 0, 0 };
    }

    std::string_view rpy_str((char *)xml_rpy);
    auto rp_break = rpy_str.find(' ');
    auto py_break = rpy_str.find(' ', rp_break + 1);

    auto r_str = rpy_str.substr(0, rp_break);
    auto p_str = rpy_str.substr(rp_break + 1, py_break - rp_break - 1);
    auto y_str = rpy_str.substr(py_break + 1);

    float r;
    std::from_chars(r_str.data(), r_str.data() + r_str.size(), r);

    auto r_quat = math::Quat::angleAxis(r, {0, 1, 0});

    float p;
    std::from_chars(p_str.data(), p_str.data() + p_str.size(), p);

    auto p_quat = math::Quat::angleAxis(r, {1, 0, 0});

    float y;
    std::from_chars(y_str.data(), y_str.data() + y_str.size(), y);

    auto y_quat = math::Quat::angleAxis(r, {0, 0, -1});

    // FIXME: order?
    auto rotation = r_quat * p_quat * y_quat;

    xmlFree(xml_rpy);

    return rotation;
}

static math::Vector3 urdfReadXYZ(xmlTextReaderPtr reader)
{
    auto xml_xyz =
        xmlTextReaderGetAttribute(reader, BAD_CAST "xyz");

    if (xml_xyz == nullptr) {
        return { 0, 0, 0 };
    }

    std::string_view xyz_str((char *)xml_xyz);
    auto xy_break = xyz_str.find(' ');
    auto yz_break = xyz_str.find(' ', xy_break + 1);

    auto x_str = xyz_str.substr(0, xy_break);
    auto y_str = xyz_str.substr(xy_break + 1, yz_break - xy_break - 1);
    auto z_str = xyz_str.substr(yz_break + 1);

    float x;
    std::from_chars(x_str.data(), x_str.data() + x_str.size(), x);

    float y;
    std::from_chars(y_str.data(), y_str.data() + y_str.size(), y);

    float z;
    std::from_chars(z_str.data(), z_str.data() + z_str.size(), z);

    xmlFree(xml_xyz);

    return { x, y, z };
}

static MergedSourceObject parseURDF(std::string_view obj_path,
                                    const math::Mat3x4 &base_txfm)
{
    xmlTextReaderPtr reader = xmlNewTextReaderFilename(
        std::string(obj_path).c_str());

    if (reader == nullptr) {
        FATAL("Failed to load URDF file: %.*s", (int)obj_path.size(),
              obj_path.data());
    }

    auto current_joint = Optional<URDFJointState>::none();
    auto current_link =
        Optional<std::pair<std::string, URDFLinkState>>::none();
    bool in_visual = false;

    std::unordered_map<std::string, URDFLinkState> links;
    std::vector<URDFJointState> joints;

    auto invalid = [&](const char *msg) {
        xmlNodePtr cur_node = xmlTextReaderCurrentNode(reader);
        int line_no = xmlGetLineNo(cur_node);

        FATAL("URDF parsing error: %s, %.*s at line %d",
              msg, (int)obj_path.size(), obj_path.data(), line_no);
    };

    auto urdfMatch = [](const xmlChar *tag, const char *name) {
        return xmlStrcmp(tag, BAD_CAST name) == 0;
    };

    auto process = [&](xmlTextReaderPtr reader) {
        auto tag_name = xmlTextReaderConstName(reader);
        int node_type = xmlTextReaderNodeType(reader);
        bool elem_start = node_type == XML_READER_TYPE_ELEMENT;
        bool elem_end = node_type == XML_READER_TYPE_END_ELEMENT;

        if (tag_name == nullptr || (!elem_start && !elem_end)) {
            return;
        }

        if (urdfMatch(tag_name, "joint")) {
            if (elem_start) {
                assert(!current_joint.has_value());

                current_joint = URDFJointState {
                    .translation = math::Vector3 { 0, 0, 0 },
                    .rotation = math::Quat { 1, 0, 0, 0 },
                    .parent = "",
                    .child = "",
                };
            } else if (elem_end) {
                if (!current_joint.has_value() ||
                        current_joint->child == "" ||
                        current_joint->parent == "") {
                    invalid("Incomplete joint");
                }

                joints.emplace_back(*current_joint);
                current_joint.reset();
            }
        } else if (urdfMatch(tag_name, "parent") && elem_start) {
            char *parent_name = (char *)
                xmlTextReaderGetAttribute(reader, BAD_CAST "link");

            if (!parent_name || !current_joint.has_value()) {
                invalid("Parent tag outside joint");
            }

            current_joint->parent = parent_name;

            xmlFree(parent_name);
        } else if (urdfMatch(tag_name, "child") && elem_start) {
            char *child_name = (char *)
                xmlTextReaderGetAttribute(reader, BAD_CAST "link");

            if (!child_name || !current_joint.has_value()) {
                invalid("Child tag outside joint");
            }

            current_joint->child = child_name;
            xmlFree(child_name);
        } else if (urdfMatch(tag_name, "link")) {
            if (elem_start) {
                char *link_name = (char *)
                    xmlTextReaderGetAttribute(reader, BAD_CAST "name");

                if (!link_name || current_link.has_value()) {
                    invalid("Link missing name");
                }

                current_link = std::pair {
                    std::string(link_name),
                    URDFLinkState {
                        .translation = { 0, 0, 0 },
                        .rotation = { 1, 0, 0, 0 },
                        .renderModel = "",
                    },
                };
            } else if (elem_end) {
                if (!current_link.has_value()) {
                    invalid("Double link closing tag");
                }

                links.emplace(std::move(*current_link));
                current_link.reset();
            }
        } else if (current_link.has_value() && urdfMatch(tag_name, "visual")) {
            if (elem_start) {
                if (in_visual) {
                    invalid("Repeat visual tag");
                }

                in_visual = true;
            }

            if (elem_end) {
                if (!in_visual) {
                    invalid("Visual closing tag without start");
                }

                in_visual = false;
            }
        } else if (urdfMatch(tag_name, "origin") && elem_start) {
            if (current_joint.has_value()) {
                current_joint->translation = urdfReadXYZ(reader);
                current_joint->rotation = urdfReadRPY(reader);
            }

            if (current_link.has_value() && in_visual) {
                current_link->second.translation = urdfReadXYZ(reader);
                current_link->second.rotation = urdfReadRPY(reader);
            }
        } else if (in_visual && urdfMatch(tag_name, "mesh") && elem_start) {
            char *filename = (char *)
                xmlTextReaderGetAttribute(reader, BAD_CAST "filename");

            if (!filename) {
                invalid("Mesh tag missing filename");
            }

            current_link->second.renderModel = filename;
            xmlFree(filename);
        }
    };

    int ret;
    while ((ret = xmlTextReaderRead(reader)) == 1) {
        process(reader);
    }

    if (ret != 0) {
        FATAL("Failed to parse URDF file: %.*s", (int)obj_path.size(),
              obj_path.data());
    }

    xmlTextReaderClose(reader);

    std::vector<URDFGraphNode> urdf_graph;
    urdf_graph.reserve(links.size());

    std::unordered_map<std::string_view, int32_t> name_to_node;

    for (const auto &[name, link] : links) {
        urdf_graph.push_back({
            .model = link.renderModel,
            .transform = math::Mat3x4::fromTRS(link.translation,
                                               link.rotation),
            .parent = -1,
            .edges = {},
        });

        name_to_node.emplace(name, int32_t(urdf_graph.size() - 1));
    }

    for (const URDFJointState &joint : joints) {
        int32_t parent_idx = name_to_node.find(joint.parent)->second;
        int32_t child_idx = name_to_node.find(joint.child)->second;

        auto &parent_node = urdf_graph[parent_idx];
        auto &child_node = urdf_graph[child_idx];

        child_node.parent = parent_idx;
        parent_node.edges.push_back({
            .transform = math::Mat3x4::fromTRS(joint.translation,
                                               joint.rotation),
            .child = child_idx,
        });
    }

    int32_t root_idx = -1;
    for (size_t i = 0; i < urdf_graph.size(); i++) {
        if (urdf_graph[i].parent == -1) {
            assert(root_idx == -1);
            root_idx = (int32_t)i;
        }
    }
    assert(root_idx != -1);

    urdf_graph[root_idx].transform =
        base_txfm.compose(urdf_graph[root_idx].transform);

    MergedSourceObject all_merged;
    std::string_view dir_name = obj_path.substr(0, obj_path.rfind('/'));

    std::vector<int32_t> node_stack;
    node_stack.reserve(urdf_graph.size());
    node_stack.push_back(root_idx);

    while (node_stack.size() > 0) {
        int32_t cur_node_idx = node_stack.back();
        node_stack.pop_back();
        const auto &cur_node = urdf_graph[cur_node_idx];

        auto cur_txfm = cur_node.transform;

        if (cur_node.model != "") {
            std::string path(dir_name);
            path += "/";
            path += cur_node.model;
            
            MergedSourceObject sub = loadAndParseGLTF(path, cur_txfm);

            for (auto &vert_list : sub.vertices) {
                all_merged.vertices.emplace_back(std::move(vert_list));
            }

            for (auto &idx_list : sub.indices) {
                all_merged.indices.emplace_back(std::move(idx_list));
            }

            for (auto &mesh : sub.meshes) {
                all_merged.meshes.emplace_back(std::move(mesh));
            }
        }

        for (const auto &edge : cur_node.edges) {
            auto &child_node = urdf_graph[edge.child];

            child_node.transform =
                cur_txfm.compose(edge.transform).compose(child_node.transform);

            node_stack.push_back(edge.child);
        }
    }

    return all_merged;
}

static int64_t parseObject(std::string_view obj_path, ParseData &parse_data,
                           bool is_urdf, const math::Mat3x4 &base_txfm)
{
    using namespace std;

    auto parsed_stages_iter =
        parse_data.parsedObjs.find(std::string(obj_path));
    if (parsed_stages_iter != parse_data.parsedObjs.end()) {
        return parsed_stages_iter->second;
    }

    std::cout << "'" << obj_path << "' " << std::endl;

    auto urdfBaseTXFM = [](math::Mat3x4 base_txfm) {
        return base_txfm.compose({{
            { 0, 0, -1 },
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 0 },
        }});
    };

    MergedSourceObject obj = is_urdf ? 
        parseURDF(obj_path, urdfBaseTXFM(base_txfm)) :
        loadAndParseGLTF(obj_path, base_txfm);

    for (auto &vert_list : obj.vertices) {
        parse_data.trainData.vertices.emplace_back(std::move(vert_list));
    }

    for (auto &idx_list : obj.indices) {
        parse_data.trainData.indices.emplace_back(std::move(idx_list));
    }

    parse_data.trainData.meshes.emplace_back(std::move(obj.meshes));

    int64_t obj_id = parse_data.trainData.objects.size();

    parse_data.trainData.objects.push_back({
        Span<const render::SourceMesh>(parse_data.trainData.meshes.back()),
    });

    auto [iter, success] = parse_data.parsedObjs.emplace(obj_path, obj_id);
    assert(success);

    return obj_id;
}

static constexpr math::Mat3x4 habitat_txfm {{
    { 1, 0, 0, },
    { 0, 0, 1, },
    { 0, -1, 0, },
    { 0, 0, 0, },
}};

static constexpr math::Mat3x4 urdf_txfm {{
    { 0, 0, 1, },
    { 0, 1, 0, },
    { -1, 0, 0, },
    { 0, 0, 0, },
}};

static constexpr math::Mat3x4 ycb_txfm {{
    { 1, 0, 0, },
    { 0, 0, 1, },
    { 0, 1, 0, },
    { 0, 0, 0, },
}};

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

    int64_t stage_obj_id =
        parseObject(stage_path, parse_data, false, habitat_txfm);

    vector<ParsedInstance> additional_instances;

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

        int64_t inst_obj_id =
            parseObject(obj_path, parse_data, false, habitat_txfm);

        math::Vector3 translation = getVec3(instance["translation"]);
        math::Quat rotation = getQuat(instance["rotation"], true);

        additional_instances.push_back({
            .objID = inst_obj_id,
            .translation = translation,
            .rotation = rotation,
            .scale = math::Vector3 { 1.f, 1.f, 1.f },
            .dynamic = false,
        });
    }

    simdjson::ondemand::array articulated_instances_arr;
    REQ_JSON(scene_json["articulated_object_instances"].get(
            articulated_instances_arr));

    auto find_urdf = [](string_view dir, string_view obj_name) {
        std::string filename = std::string(obj_name) + ".urdf";

        for (const auto &dir_entry :
             std::filesystem::recursive_directory_iterator(dir)) {
            if (!dir_entry.is_regular_file()) {
                continue;
            }

            const std::filesystem::path &path = dir_entry.path();

            if (path.filename() == filename) {
                return path.string();
            }
        }

        FATAL("Could not find URDF for %s", obj_name);
    };

    std::string urdf_dir = parse_data.dataDir;
    urdf_dir += "replica_cad/urdf/";
    for (auto instance : articulated_instances_arr) {
        string_view obj_name;
        REQ_JSON(instance["template_name"].get(obj_name));

        math::Vector3 translation = getVec3(instance["translation"]);
        math::Quat rotation = getQuat(instance["rotation"], true);

        double uniform_scale;
        auto scale_err = instance["uniform_scale"].get(uniform_scale);
        if (scale_err) {
            uniform_scale = 1.0;
        }

        math::Vector3 scale {
            (float)uniform_scale,
            (float)uniform_scale,
            (float)uniform_scale,
        };

        std::string urdf_path = find_urdf(urdf_dir, obj_name);

        int64_t inst_obj_id =
            parseObject(urdf_path, parse_data, true, urdf_txfm);

        additional_instances.push_back({
            .objID = inst_obj_id,
            .translation = translation,
            .rotation = rotation,
            .scale = scale,
            .dynamic = false,
        });
    }

    ParsedScene parsed_scene {
        .stageObjID = stage_obj_id,
        .stageTranslation = stage_translation,
        .stageRotation = stage_rotation,
        .additionalInstances = std::move(additional_instances),
    };

    auto [iter, success] =
        parse_data.parsedScenes.emplace(scene_path, std::move(parsed_scene));
    assert(success);

    return &iter->second;
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
        math::Quat start_rot = getQuat(episode["start_rotation"], false);

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
            auto obj_prefix =
                obj_config_path.substr(0, obj_config_path.find('.'));
            obj_path += obj_prefix;
            obj_path += "/google_16k/textured.glb";

            int64_t inst_obj_id =
                parseObject(obj_path, parse_data, false, ycb_txfm);

            dyn_insts.push_back(ParsedInstance {
                .objID = inst_obj_id,
                .translation = translation,
                .rotation = rot,
                .scale = scale,
                .dynamic = true,
            });
        }

        auto fixTranslation = [](math::Vector3 v) {
            return math::Vector3 { v.x, -v.z, v.y };
        };

        auto fixRotation = [](math::Quat q) {
            return math::Quat { q.w, q.x, -q.z, q.y };
        };

        auto fixScale = [](math::Vector3 s) {
            return math::Vector3 { s.x, s.z, s.y };
        };

        int32_t inst_offset = parse_data.trainData.instances.size();
        parse_data.trainData.instances.push_back({
            int32_t(scene->stageObjID),
            fixTranslation(scene->stageTranslation),
            fixRotation(scene->stageRotation),
            math::Vector3 { 1.f, 1.f, 1.f },
        });

        for (const ParsedInstance &inst : scene->additionalInstances) {
            parse_data.trainData.instances.push_back({
                int32_t(inst.objID),
                fixTranslation(inst.translation),
                fixRotation(inst.rotation),
                fixScale(inst.scale),
            });
        }

        int32_t dyn_offset = parse_data.trainData.instances.size();

        for (const ParsedInstance &inst : dyn_insts) {
            parse_data.trainData.instances.push_back({
                int32_t(inst.objID),
                fixTranslation(inst.translation),
                fixRotation(inst.rotation),
                fixScale(inst.scale),
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
            fixTranslation(start_pos),
            fixRotation(start_rot),
            fixTranslation(dyn_insts[0].translation),
        });
    }

    std::cout << parse_data.trainData.episodes.size() << std::endl;
    return std::move(parse_data.trainData);
}

}
