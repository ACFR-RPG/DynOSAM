#pragma once

#include <gtsam/inference/Key.h>

#include <optional>

#include "dynosam_common/Exceptions.hpp"

namespace dyno {

/// @brief Status for a StateQuery<> type. This is defined outside the
/// StateQuery class so that the type of status is independant of the StateQuery
/// template.
enum StateQueryStatus { VALID, NOT_IN_MAP, WAS_IN_MAP, INVALID_MAP };
/**
 * @brief Represents an optional value with meta-data that is retrieved from the
 * map.
 *
 * @tparam ValueType A value type stored in gtsam::Values
 */
template <typename ValueType>
class StateQuery : public std::optional<ValueType> {
 public:
  //! Base type representing the existance of the query value
  using Base = std::optional<ValueType>;

  //! Status of the state query
  using Status = StateQueryStatus;
  gtsam::Key key_;
  Status status_;

  inline gtsam::Key key() const { return key_; }
  inline Status status() const { return status_; }

  StateQuery() {}
  StateQuery(gtsam::Key key, const ValueType& v) : key_(key), status_(VALID) {
    Base::emplace(v);
  }
  StateQuery(gtsam::Key key, Status status) : key_(key), status_(status) {}

  const ValueType& get() const {
    if (!Base::has_value()) {
      DYNO_THROW_MSG(DynosamException)
          << "StateQuery has no value for query type "
          << type_name<ValueType>() + " with key " << DynosamKeyFormatter(key_)
          << " and status " << std::to_string(status());
    }
    return Base::value();
  }

  ValueType getOr(const ValueType& default_value) const {
    return Base::value_or(default_value);
  }

  bool isValid() const { return status_ == VALID; }

  static StateQuery InvalidMap() {
    return StateQuery(gtsam::Key{}, INVALID_MAP);
  }
  static StateQuery NotInMap(gtsam::Key key) {
    return StateQuery(key, NOT_IN_MAP);
  }
  static StateQuery WasInMap(gtsam::Key key) {
    return StateQuery(key, WAS_IN_MAP);
  }

 private:
  using Base::value;
  using Base::value_or;
};

/**
 * @brief Safe getter to a StateQuery object.
 * If the StateQuery is successful, result is set to the value of the query and
 * true is returned. Else, result is set to to the default value and false is
 * returned.
 *
 * @tparam ValueType
 * @param result
 * @param query
 * @param default_value
 * @return true
 * @return false
 */
template <typename ValueType>
bool getSafeQuery(ValueType& result, const StateQuery<ValueType>& query,
                  const ValueType& default_value) {
  if (query) {
    result = query.get();
    return true;
  } else {
    result = default_value;
    return false;
  }
}

namespace internal {

template <typename T>
T getStateQueryDebugHelper(const StateQuery<T>& query, const char* file,
                           int line) {
  try {
    return query.get();
  } catch (const DynosamException& e) {
    throw DynosamExceptionDebug(e.what(), file, line);
  }
}

}  // namespace internal

}  // namespace dyno

#define DYNO_GET_QUERY_DEBUG(state_query) \
  dyno::internal::getStateQueryDebugHelper(state_query, __FILE__, __LINE__)
