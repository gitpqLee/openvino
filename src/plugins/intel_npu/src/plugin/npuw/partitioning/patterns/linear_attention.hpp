// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace npuw {
namespace online {
class Snapshot;
}  // namespace online
namespace patterns {
namespace linear_attn {

class LinearAttention : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::linear_attn::LinearAttention");
    static constexpr const char* pattern_name() {
        return "LinearAttention";
    }
    static constexpr const char* isolation_tag() {
        return "linear_attn";
    }
    static constexpr const char* group_name() {
        return "linear_attn";
    }
    LinearAttention(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace linear_attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
