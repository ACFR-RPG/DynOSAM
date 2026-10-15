#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

namespace dyno2 {

typedef std::size_t Index;
typedef std::pair<Index, Index> IndexMatch;
typedef std::vector<IndexMatch> IndexMatches;
typedef std::unordered_map<Index, Index> IndexMapping;

}  // namespace dyno2
