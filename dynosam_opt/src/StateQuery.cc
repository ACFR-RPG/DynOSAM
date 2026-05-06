#include "dynosam_opt/StateQuery.hpp"

namespace dyno {

template <>
std::string to_string(const StateQueryStatus& status) {
  switch (status) {
    case StateQueryStatus::VALID:
      return "VALID";
    case StateQueryStatus::NOT_IN_MAP:
      return "NOT_IN_MAP";
    case StateQueryStatus::WAS_IN_MAP:
      return "WAS_IN_MAP";
    case StateQueryStatus::INVALID_MAP:
      return "INVALID_MAP";
    default:
      return "Unknown";
  }
}

}  // namespace dyno
