find_package(Boost COMPONENTS python3 numpy3 REQUIRED)
list(APPEND dynosam_nn_LIBRARIES ${Boost_LIBRARIES})
