/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <glog/logging.h>

#include <cmath>
#include <opencv2/viz/types.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Numerical.hpp"

namespace dyno {

// Encapsulated differences between processing float/double depths and uint8_t
template <typename T, typename Enable = void>
struct color_traits;

template <typename T>
struct _floating_point_colour_traits {
  static constexpr inline T max() { return 1.0; }
};

template <typename T>
struct _int_point_colour_traits {
  static constexpr inline T max() { return 255; }
};

// Specalisation for integer types
template <typename T>
struct color_traits<T, std::enable_if_t<std::is_integral<T>::value>>
    : _int_point_colour_traits<T> {};
// Specalisation for floating point types
template <typename T>
struct color_traits<T, std::enable_if_t<std::is_floating_point<T>::value>>
    : _floating_point_colour_traits<T> {};

template <typename T>
struct RGBA {
  // Data.
  T r = 0;
  T g = 0;
  T b = 0;
  T a = max();

  RGBA() = default;
  RGBA(T r, T g, T b, T a = max()) : r(r), g(g), b(b), a(a) {}

  template <typename U>
  RGBA(const RGBA<U>& other) {
    // check if we need to go up (1 -> 255) or down (255 -> 1)
    constexpr T other_max = static_cast<T>(RGBA<U>::max());

    // this is lower than other so we need to normalize using other max
    if constexpr (this->max() < other_max) {
      this->r = is_zero(other.r) ? 0 : static_cast<T>(other.r) / other_max;
      this->g = is_zero(other.g) ? 0 : static_cast<T>(other.g) / other_max;
      this->b = is_zero(other.b) ? 0 : static_cast<T>(other.b) / other_max;
      this->a = is_zero(other.a) ? 0 : static_cast<T>(other.a) / other_max;
      CHECK_LE(this->a, max());
    }
    // this is higher than other so we need to normalize using this max
    else {
      constexpr U cast_max = static_cast<U>(max());
      this->r = is_zero(other.r) ? 0 : static_cast<T>(other.r * cast_max);
      this->g = is_zero(other.g) ? 0 : static_cast<T>(other.g * cast_max);
      this->b = is_zero(other.b) ? 0 : static_cast<T>(other.b * cast_max);
      this->a = is_zero(other.a) ? 0 : static_cast<T>(other.a * cast_max);
      CHECK_LE(this->a, max());
    }
  }

  //! Maximum value (upper range) for the RGBA values given the type
  //! Should be 255u for integer types and 1.0f for floating point types
  static constexpr inline T max() { return color_traits<T>::max(); }
};

// From
// https://github.com/MIT-SPARK/Spark-DSG/blob/main/include/spark_dsg/color.h
struct Color : RGBA<uint8_t> {
  using RGBA<uint8_t>::max;

  // Constructors.
  Color() = default;
  Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = max())
      : RGBA(r, g, b, a) {}
  virtual ~Color() = default;

  // Operators.
  bool operator==(const Color& other) const;
  bool operator!=(const Color& other) const { return !(*this == other); }
  bool operator<(const Color& other) const;
  bool operator>(const Color& other) const { return other < *this; }
  friend std::ostream& operator<<(std::ostream&, const Color&);

  // Operator to cv::Scalar in RGBA format
  operator cv::Scalar() const { return rgba(); }

  /**
   * @brief Return in rgba format
   *
   * @return cv::Scalar
   */
  inline cv::Scalar rgba() const { return cv::Scalar(r, g, b, a); }

  /**
   * @brief Return in BGRA format.
   *
   * @return cv::Scalar
   */
  inline cv::Scalar bgra() const { return cv::Scalar(b, g, r, a); }

  /**
   * @brief Return the RGBA values as a double. Note this will change the range
   * from 0-255 to 0-1.
   *
   * @return RGBA<double>
   */
  inline RGBA<double> toDouble() const { return RGBA<double>(*this); }

  // Tools.
  /**
   * @brief Blend another color with this one.
   * @param other The other color to blend.
   * @param weight The weight of the other color [0,1].
   * @return The resulting blended color.
   */
  Color blend(const Color& other, float weight = 0.5f) const;

  /**
   * @brief Merge another color into this one.
   * @param other The other color to merge.
   * @param weight The weight of the other color [0,1].
   */
  void merge(const Color& other, float weight = 0.5f);

  /**
   * @brief Hash function for colors to use colors as keys in maps, sets, etc.
   */
  struct Hash {
    uint32_t operator()(const Color& color) const {
      return color.r + (color.g << 8) + (color.b << 16) + (color.a << 24);
    }
  };

  // Presets.
  static Color black() { return Color(0, 0, 0); }
  static Color white() { return Color(255, 255, 255); }
  static Color red() { return Color(255, 0, 0); }
  static Color green() { return Color(0, 255, 0); }
  static Color blue() { return Color(0, 0, 255); }
  static Color yellow() { return Color(255, 255, 0); }
  static Color orange() { return Color(255, 127, 0); }
  static Color purple() { return Color(127, 0, 255); }
  static Color cyan() { return Color(0, 255, 255); }
  static Color magenta() { return Color(255, 0, 255); }
  static Color pink() { return Color(255, 127, 127); }
  static Color random();

  // Conversions.
  static Color fromHSV(float hue, float saturation, float value);
  static Color fromHLS(float hue, float luminance, float saturation);

  // Color maps.
  /**
   * @brief Generate a gray-scale color.
   * @param value The value of the gray [0,1, where 0 is black and 1 is white].
   */
  static Color gray(float value = 0.5f);

  /**
   * @brief Generate a color based on a quality value.
   * @param value The quality value [0,1], where 0 is red, 0.5 is yellow, and 1
   * is green.
   */
  static Color quality(float value);

  /**
   * @brief Generate a color based on a spectrum of colors.
   * @param value The spectrum value [0,1], where 0 is the first color and 1 is
   * the last color.
   * @param colors The list of colors in the spectrum. Colors are assumed
   * equidistant.
   */
  static Color spectrum(float value, const std::vector<Color>& colors);

  /**
   * @brief Generate a color based on a ironbow value.
   * @param value The temperature value [0,1], where 0 is dark and 1 is light.
   */
  static Color ironbow(float value);

  /**
   * @brief Generate a color based on a rainbow value.
   * @param value The rainbow value [0,1], where 0 is red, 0.5 is green, and 1
   * is blue.
   */
  static Color rainbow(float value);

  /**
   * @brief Generate a sequence of never repeating colors in the rainbow
   * spectrum.
   * @param id The id of the color in the sequence.
   * @param ids_per_revolution The number of colors per revolution of the hue.
   */
  static Color rainbowId(size_t id, size_t ids_per_revolution = 16);

  static constexpr float unique_id_default_value = 0.95f;
  static constexpr float unique_id_default_saturation = 0.5f;

  /**
   * @brief Generates a sequence of never repeating colors using the golden
   * ratio to seperate the colours along the HSV ring spectrum.
   *
   * @param id The id of the color in the sequence.
   * @param saturation Saturation value
   * @param value Value value ;)
   * @return Color
   */
  static Color uniqueId(size_t id,
                        float saturation = unique_id_default_saturation,
                        float value = unique_id_default_value);

  /**
   * @brief Overload of uniqueId that uses types specific for ObjectId.
   * To allow a simpler function signature, this function only takes one
   * argument where the saturation=0.5f and value=0.95f.
   *
   * @param id ObjectId
   * @return Color
   */
  static Color uniqueObjectId(ObjectId id) {
    return uniqueId(static_cast<size_t>(id));
  }

 private:
  static const std::vector<Color> ironbow_colors_;
};

/**
 * @brief Palette of colours based of 'color blindness' which presents a set of
 * colours which are optimized for color-blinded individuals. In other words,
 * they are very easy to see and look very nice. To be used in vizualizations.
 *
 */
struct NiceColors {
  static Color black() { return Color(0, 0, 0); }
  static Color orange() { return Color(230, 159, 0); }
  static Color skyblue() { return Color(86, 180, 233); }
  static Color bluishgreen() { return Color(0, 158, 115); }
  static Color yellow() { return Color(240, 228, 66); }
  static Color blue() { return Color(0, 114, 178); }
  static Color vermillion() { return Color(213, 94, 0); }
  static Color reddishpurple() { return Color(204, 121, 167); }
};

}  // namespace dyno
