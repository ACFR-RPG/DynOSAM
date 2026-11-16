macro(detect_ros2_version)
    # ----------------------------------------------------------------------
    # ROS 2 distribution detection
    #
    # Primary: ROS_DISTRO environment variable.
    # Fallback: Installed rclcpp package version.
    #
    # Humble:  rclcpp 16.x
    # Jazzy:   rclcpp 28.x
    # Kilted:  rclcpp 29.x
    # ----------------------------------------------------------------------

    set(ROS2_AT_LEAST_JAZZY OFF)

    if(DEFINED ENV{ROS_DISTRO} AND NOT "$ENV{ROS_DISTRO}" STREQUAL "")
        set(ROS_DISTRO_DETECTED "$ENV{ROS_DISTRO}")

        message(STATUS "Detected ROS 2 distribution: ${ROS_DISTRO_DETECTED}")

        if(ROS_DISTRO_DETECTED STREQUAL "jazzy" OR
           ROS_DISTRO_DETECTED STREQUAL "kilted" OR
           ROS_DISTRO_DETECTED STREQUAL "rolling")
            set(ROS2_AT_LEAST_JAZZY ON)

        elseif(ROS_DISTRO_DETECTED STREQUAL "humble" OR
               ROS_DISTRO_DETECTED STREQUAL "iron")
            set(ROS2_AT_LEAST_JAZZY OFF)

        else()
            message(WARNING
                "Unknown ROS_DISTRO='${ROS_DISTRO_DETECTED}'. "
                "Falling back to rclcpp version (${rclcpp_VERSION})."
            )

            if(rclcpp_VERSION VERSION_GREATER_EQUAL "28.0.0")
                set(ROS2_AT_LEAST_JAZZY ON)
            endif()
        endif()

    else()
        message(WARNING
            "ROS_DISTRO environment variable is not defined. "
            "Falling back to rclcpp version (${rclcpp_VERSION})."
        )

        if(rclcpp_VERSION VERSION_GREATER_EQUAL "28.0.0")
            set(ROS2_AT_LEAST_JAZZY ON)
        endif()
    endif()

    # ----------------------------------------------------------------------
    # Export compile definition
    # ----------------------------------------------------------------------

    if(ROS2_AT_LEAST_JAZZY)
        message(STATUS "ROS 2 compatibility: AT LEAST JAZZY")
        add_compile_definitions(ROS2_AT_LEAST_JAZZY=1)
    else()
        message(STATUS "ROS 2 compatibility: OLDER THAN JAZZY")
        add_compile_definitions(ROS2_AT_LEAST_JAZZY=0)
    endif()

endmacro()
