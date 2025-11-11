# cmake-lint: disable=C0103 find_package file based on
# https://izzys.casa/2020/12/how-to-find-packages-with-cmake-the-basics/ and
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#find-modules
#
#[===============================================================================[.rst:
FindTensorRT
------------

Finds TensorRT and related component libraries.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``TensorRT::TensorRT``
  The TensorRT library and selected components

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``TensorRT_FOUND``
  True if the system has TensorRT
``TensorRT_VERSION``
  The version of TensorRT that was found
``TensorRT_INCLUDE_DIRS``
  The directories containing headers for TensorRT
``TensorRT_LIBRARIES``
  Selected libraries of TensorRT

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set
``TensorRT_INCLUDE_DIR``
  The directory containing ``NvInfer.h``
``TensorRT_LIBRARY``
  The path to the nvinfer library

#]===============================================================================]
include(FindPackageHandleStandardArgs)

# find actual location of TensorRT
find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h)
find_library(TensorRT_LIBRARY NAMES nvinfer)

# Get TensorRT version from header
if(TensorRT_INCLUDE_DIR)
  file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" version-file)
  if(NOT version-file)
    message(AUTHOR_WARNING "TensorRT version is missing")
  endif()

  set(VERSION_ITEMS "")
  foreach(arg IN ITEMS MAJOR MINOR PATCH BUILD)
    set(REGEX_STRING "#define NV_TENSORRT_${arg} ([0-9]+)[^\n]*\n")
    string(REGEX MATCH "${REGEX_STRING}" line-match ${version-file})
    if(line-match)
      list(APPEND VERSION_ITEMS ${CMAKE_MATCH_1})
    endif()
  endforeach()

  list(JOIN VERSION_ITEMS "." TensorRT_VERSION)
endif()

# Find all requested components
foreach(component IN LISTS TensorRT_FIND_COMPONENTS)
  find_library(TensorRT_${component}_LIBRARY NAMES "nv${component}")
  if(TensorRT_${component}_LIBRARY)
    set(TensorRT_${component}_FOUND TRUE)
  endif()
endforeach()

find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
  VERSION_VAR TensorRT_VERSION
  HANDLE_COMPONENTS
)

if(TensorRT_FOUND)
  mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)
  set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
endif()

set(TensorRT_COMPONENT_LIBRARIES "")
foreach(component IN LISTS TensorRT_FIND_COMPONENTS)
  if(TensorRT_${component}_FOUND AND NOT TARGET TensorRT::${component})
    add_library(TensorRT::${component} UNKNOWN IMPORTED)
    set_target_properties(
      TensorRT::${component} PROPERTIES IMPORTED_LOCATION "${TensorRT_${component}_LIBRARY}"
                                        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
    list(APPEND TensorRT_COMPONENT_LIBRARIES TensorRT::${component})
  endif()
endforeach()

if(TensorRT_FOUND AND NOT TARGET TensorRT::TensorRT)
  add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
  set_target_properties(
    TensorRT::TensorRT
    PROPERTIES IMPORTED_LOCATION "${TensorRT_LIBRARY}"
               VERSION "${TensorRT_VERSION}"
               INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${TensorRT_COMPONENT_LIBRARIES}"
  )
endif()
