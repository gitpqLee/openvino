// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "linear_attention.hpp"

#include <deque>
#include <unordered_set>

#include "../../logging.hpp"
#include "../online/group.hpp"
#include "../online/snapshot.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace linear_attn {

namespace {
namespace opp = ov::pass::pattern;

using NodeSet = std::unordered_set<std::shared_ptr<ov::Node>>;

void collect_to_projection(const std::shared_ptr<ov::Node>& root,
                           const std::shared_ptr<ov::Node>& recurrent_core,
                           NodeSet& nodes) {
    std::deque<std::shared_ptr<ov::Node>> pending{root};
    while (!pending.empty()) {
        auto node = pending.front();
        pending.pop_front();
        if (!node || !nodes.insert(node).second || node == recurrent_core) {
            continue;
        }
        if (ov::is_type<ov::op::v0::MatMul>(node)) {
            continue;
        }
        for (const auto& input : node->inputs()) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (ov::npuw::online::detail::isOp(parent)) {
                pending.push_back(std::move(parent));
            }
        }
    }
}

std::shared_ptr<ov::Node> find_output_projection(const ov::Output<ov::Node>& core_output) {
    std::deque<std::shared_ptr<ov::Node>> pending;
    NodeSet visited;
    for (const auto& input : core_output.get_target_inputs()) {
        pending.push_back(input.get_node()->shared_from_this());
    }

    while (!pending.empty()) {
        auto node = pending.front();
        pending.pop_front();
        if (!visited.insert(node).second) {
            continue;
        }
        if (ov::is_type<ov::op::v0::MatMul>(node)) {
            return node;
        }
        for (const auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                pending.push_back(input.get_node()->shared_from_this());
            }
        }
    }
    return {};
}

void collect_connected_state_branch(const std::shared_ptr<ov::Node>& assign, NodeSet& mixer_nodes) {
    std::deque<std::shared_ptr<ov::Node>> pending{assign};
    NodeSet branch;
    while (!pending.empty()) {
        auto node = pending.front();
        pending.pop_front();
        if (mixer_nodes.count(node)) {
            mixer_nodes.insert(branch.begin(), branch.end());
            return;
        }
        if (!node || !branch.insert(node).second || ov::is_type<ov::op::v0::MatMul>(node)) {
            continue;
        }
        for (const auto& input : node->inputs()) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (ov::npuw::online::detail::isOp(parent)) {
                pending.push_back(std::move(parent));
            }
        }
    }
}
}  // namespace

LinearAttention::LinearAttention(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                 const std::string& isol_tag) {
    auto recurrent_core = opp::wrap_type<ov::op::v5::Loop>([](const ov::Output<ov::Node>& output) {
        return ov::pass::matches_gated_delta_net_loop(output.get_node_shared_ptr());
    });
    auto node_to_group = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& matcher) {
        const auto core = matcher.get_match_root();
        auto loop = ov::as_type_ptr<ov::op::v5::Loop>(core);
        auto output_projection = find_output_projection(loop->output(0));
        if (!output_projection) {
            return false;
        }

        LOG_DEBUG("LinearAttention pattern matched at " << core->get_friendly_name());

        NodeSet mixer_nodes{core};
        for (size_t input_idx = 2; input_idx <= 6; ++input_idx) {
            collect_to_projection(loop->get_input_node_shared_ptr(input_idx), core, mixer_nodes);
        }
        mixer_nodes.insert(output_projection);
        collect_to_projection(output_projection->get_input_node_shared_ptr(0), core, mixer_nodes);
        for (const auto& [node, group] : *node_to_group) {
            if (ov::is_type<ov::op::v6::Assign>(node)) {
                collect_connected_state_branch(node, mixer_nodes);
            }
        }

        for (const auto& node : mixer_nodes) {
            if (const auto group = node_to_group->find(node); group != node_to_group->end()) {
                group->second->isolate(isol_tag);
            }
        }

        return false;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(recurrent_core, pattern_name());
    register_matcher(matcher, callback);
}

}  // namespace linear_attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
