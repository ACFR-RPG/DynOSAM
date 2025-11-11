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

#include "dynosam_common/viz/Colour.hpp"

#include <algorithm>
#include <random>

namespace dyno {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<uint8_t> dis(0, 255);

bool Color::operator==(const Color& other) const {
  return r == other.r && g == other.g && b == other.b && a == other.a;
}

bool Color::operator<(const Color& other) const {
  if (r != other.r) {
    return r < other.r;
  }
  if (g != other.g) {
    return g < other.g;
  }
  if (b != other.b) {
    return b < other.b;
  }
  return a < other.a;
}

void Color::merge(const Color& other, float weight) {
  *this = blend(other, weight);
}

Color Color::blend(const Color& other, float weight) const {
  weight = std::clamp(weight, 0.0f, 1.0f);
  return Color(static_cast<uint8_t>(r * (1.0f - weight) + other.r * weight),
               static_cast<uint8_t>(g * (1.0f - weight) + other.g * weight),
               static_cast<uint8_t>(b * (1.0f - weight) + other.b * weight),
               static_cast<uint8_t>(a * (1.0f - weight) + other.a * weight));
}

Color Color::random() { return Color(dis(gen), dis(gen), dis(gen), dis(gen)); }

std::array<float, 3> rgbFromChromaHue(float chroma, float hue) {
  const float hue_prime = hue * 6.0f;
  const float x = chroma * (1.0f - std::abs(std::fmod(hue_prime, 2) - 1.0f));
  if (hue_prime < 1) {
    return {chroma, x, 0};
  } else if (hue_prime < 2) {
    return {x, chroma, 0};
  } else if (hue_prime < 3) {
    return {0, chroma, x};
  } else if (hue_prime < 4) {
    return {0, x, chroma};
  } else if (hue_prime < 5) {
    return {x, 0, chroma};
  }
  return {chroma, 0, x};
}

Color Color::fromHSV(float hue, float saturation, float value) {
  hue = std::clamp(hue, 0.0f, 1.0f);
  saturation = std::clamp(saturation, 0.0f, 1.0f);
  value = std::clamp(value, 0.0f, 1.0f);
  const float chroma = value * saturation;
  const auto [r1, g1, b1] = rgbFromChromaHue(chroma, hue);
  const float m = value - chroma;
  return Color(static_cast<uint8_t>((r1 + m) * 255),
               static_cast<uint8_t>((g1 + m) * 255),
               static_cast<uint8_t>((b1 + m) * 255));
}

Color Color::fromHLS(float hue, float luminance, float saturation) {
  hue = std::clamp(hue, 0.0f, 1.0f);
  luminance = std::clamp(luminance, 0.0f, 1.0f);
  saturation = std::clamp(saturation, 0.0f, 1.0f);
  const float chroma = (1.0f - std::abs(2.0f * luminance - 1.0f)) * saturation;
  const auto [r1, g1, b1] = rgbFromChromaHue(chroma, hue);
  const float m = luminance - chroma / 2.0f;
  return Color(static_cast<uint8_t>((r1 + m) * 255),
               static_cast<uint8_t>((g1 + m) * 255),
               static_cast<uint8_t>((b1 + m) * 255));
}

Color Color::gray(float value) {
  value = std::clamp(value, 0.0f, 1.0f);
  return Color(static_cast<uint8_t>(value * 255),
               static_cast<uint8_t>(value * 255),
               static_cast<uint8_t>(value * 255));
}

Color Color::quality(float value) {
  value = std::clamp(value, 0.0f, 1.0f);
  Color color;
  if (value > 0.5f) {
    color.r = (1.f - value) * 2 * 255;
    color.g = 255;
  } else {
    color.r = 255;
    color.g = value * 2 * 255;
  }
  return color;
}

Color Color::spectrum(float value, const std::vector<Color>& colors) {
  if (colors.empty()) {
    return Color::black();
  }
  value = std::clamp(value, 0.0f, 1.0f);
  const size_t num_steps = colors.size() - 1;
  const size_t index = static_cast<size_t>(value * num_steps);
  if (index >= num_steps) {
    return colors.at(num_steps);
  }
  const float weight = value * num_steps - index;
  return colors.at(index).blend(colors.at(index + 1), weight);
}

Color Color::ironbow(float value) { return spectrum(value, ironbow_colors_); }

Color Color::rainbow(float value) {
  return fromHLS(std::clamp(value, 0.0f, 1.0f), 0.5f, 1.0f);
}

/**
 * @brief Map a potentially infinite number of ids to a never repeating pattern
 * in [0, 1].
 */
float exponentialOffsetId(size_t id, size_t ids_per_revolution) {
  const size_t revolution = id / ids_per_revolution;
  const float progress_along_revolution =
      std::fmod(static_cast<float>(id) / ids_per_revolution, 1.f);
  float offset = 0.0f;
  if (ids_per_revolution < id + 1u) {
    const size_t current_episode = std::floor(std::log2(revolution));
    const size_t episode_start = std::exp2(current_episode);
    const size_t current_subdivision = revolution - episode_start;
    const float subdivision_step_size =
        1.0f / (ids_per_revolution * 2 * episode_start);
    offset = (2.0f * current_subdivision + 1) * subdivision_step_size;
  }
  return progress_along_revolution + offset;
}

Color Color::rainbowId(size_t id, size_t ids_per_revolution) {
  return Color::rainbow(exponentialOffsetId(id, ids_per_revolution));
}

Color Color::uniqueId(size_t id, float saturation, float value) {
  constexpr static float phi = (1 + std::sqrt(5.0)) / 2.0;
  auto n = (float)id * phi - std::floor((float)id * phi);

  // want in 255 mode
  float hue = std::floor(n * 256.0) / 255.0;
  return Color::fromHSV(hue, saturation, value);
}

const std::vector<Color> Color::ironbow_colors_ = {
    {0, 0, 0}, {145, 20, 145}, {255, 138, 0}, {255, 230, 40}, {255, 255, 255}};

std::ostream& operator<<(std::ostream& out, const Color& color) {
  out << "[RGBA: " << static_cast<int>(color.r) << ", "
      << static_cast<int>(color.g) << ", " << static_cast<int>(color.b) << ", "
      << static_cast<int>(color.a) << "]";
  return out;
}

}  // namespace dyno
