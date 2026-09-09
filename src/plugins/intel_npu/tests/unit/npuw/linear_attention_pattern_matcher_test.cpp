// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "partitioning/online/compiler.hpp"
#include "partitioning/online/group.hpp"
#include "partitioning/online/snapshot.hpp"

namespace {

::intel_npu::Config make_cfg(const ::intel_npu::Config::ConfigMap& cfg_map) {
    auto options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::registerNPUWOptions(*options);
    auto config = ::intel_npu::Config(options);
    config.update(cfg_map);
    return config;
}

void make_stateless_and_static(const std::shared_ptr<ov::Model>& model) {
    ov::pass::StatefulToStateless().run_on_model(model);

    std::map<std::string, ov::PartialShape> input_shapes;
    for (const auto& input : model->inputs()) {
        auto shape = input.get_partial_shape();
        for (auto& dim : shape) {
            if (dim.is_dynamic()) {
                dim = 1;
            }
        }
        input_shapes.emplace(input.get_any_name(), std::move(shape));
    }
    model->reshape(input_shapes);
}

struct OpaqueModel {
    std::shared_ptr<ov::Model> model;
    std::unordered_set<std::string> recurrent_cores;
    std::unordered_set<std::string> excluded_ops;
};

OpaqueModel build_opaque_gdn_model() {
    OpaqueModel opaque{ov::test::npuw::build_hybrid_llm_test_model(), {}, {}};
    size_t node_idx = 0;
    for (const auto& node : opaque.model->get_ordered_ops()) {
        const auto original_name = node->get_friendly_name();
        const auto opaque_name = "op_" + std::to_string(node_idx++);
        if (ov::is_type<ov::op::v5::Loop>(node)) {
            opaque.recurrent_cores.insert(opaque_name);
        }
        if (original_name.find(".mlp.") != std::string::npos ||
            original_name.find("post_attention_residual") != std::string::npos) {
            opaque.excluded_ops.insert(opaque_name);
        }
        node->set_friendly_name(opaque_name);
    }
    return opaque;
}

std::unordered_set<std::string> get_tagged_ops(const std::shared_ptr<ov::Model>& model) {
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    ov::npuw::online::PassContext context;
    context.isolates = {
        {ov::npuw::online::PatternType::PATTERN, "LinearAttention", "linear_attn"},
    };
    snapshot->setCtx(context);
    snapshot->buildGraph();
    snapshot->earlyRegroup();

    std::unordered_set<std::string> tagged_ops;
    for (const auto& [node, group] : *snapshot->getNodeToGroupMap()) {
        if (group->isolatedTag() == "linear_attn") {
            tagged_ops.insert(node->get_friendly_name());
        }
    }
    return tagged_ops;
}

}  // namespace

TEST(LinearAttentionPatternMatcherTest, MatchesRenamedGdnTopologyWithoutMlpOrResidual) {
    auto opaque = build_opaque_gdn_model();
    const auto tagged_ops = get_tagged_ops(opaque.model);

    EXPECT_FALSE(tagged_ops.empty());
    EXPECT_EQ(opaque.recurrent_cores.size(), 2u);
    for (const auto& name : opaque.recurrent_cores) {
        EXPECT_EQ(tagged_ops.count(name), 1u) << name;
    }
    const auto ordered_ops = opaque.model->get_ordered_ops();
    const auto tagged_assigns = std::count_if(ordered_ops.begin(),
                                              ordered_ops.end(),
                                              [&](const std::shared_ptr<ov::Node>& node) {
                                                  return ov::is_type<ov::op::v6::Assign>(node) &&
                                                         tagged_ops.count(node->get_friendly_name()) != 0;
                                              });
    EXPECT_EQ(tagged_assigns, 4);
    for (const auto& name : opaque.excluded_ops) {
        EXPECT_EQ(tagged_ops.count(name), 0u) << name;
    }
}

TEST(LinearAttentionPatternMatcherTest, RejectsMalformedGdnLoopBody) {
    auto opaque = build_opaque_gdn_model();
    for (const auto& node : opaque.model->get_ordered_ops()) {
        auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
        if (!loop) {
            continue;
        }
        for (const auto& body_node : loop->get_function()->get_ordered_ops()) {
            if (ov::is_type<ov::op::v0::Exp>(body_node)) {
                ov::replace_node(body_node, std::make_shared<ov::op::v0::Abs>(body_node->input_value(0)));
                break;
            }
        }
    }

    EXPECT_TRUE(get_tagged_ops(opaque.model).empty());
}

TEST(LinearAttentionPatternMatcherTest, CompletesRepPartitioningForStatelessModel) {
    auto model = ov::test::npuw::build_hybrid_llm_test_model();
    make_stateless_and_static(model);

    auto config = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"},
                            {"NPUW_ONLINE_ISOLATE", "ATTN,LINEAR_ATTN"},
                            {"NPUW_ONLINE_KEEP_BLOCKS_TAGGED", "attn,linear_attn"}});
    EXPECT_NO_THROW(ov::npuw::online::buildPartitioning(model, config));
}

TEST(LinearAttentionPatternMatcherTest, RegPipelineHonorsLinearAttentionIsolation) {
    auto model = ov::test::npuw::build_hybrid_llm_test_model();
    make_stateless_and_static(model);
    auto config = make_cfg({{"NPUW_ONLINE_PIPELINE", "REG"},
                            {"NPUW_ONLINE_ISOLATE", "LINEAR_ATTN"},
                            {"NPUW_ONLINE_KEEP_BLOCKS_TAGGED", "linear_attn"}});

    const auto partitioning = ov::npuw::online::buildPartitioning(model, config);
    EXPECT_TRUE(std::any_of(partitioning.groups.begin(), partitioning.groups.end(), [](const auto& group) {
        return group.gettag() == "linear_attn";
    }));
}
