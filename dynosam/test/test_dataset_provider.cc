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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <exception>

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "internal/tmp_file.hpp"

double randomGetItem(size_t idx) { return idx + 3.5; }

class DoubleDataFolder : public dyno::DataFolder<double> {
 public:
  DoubleDataFolder() {}

  std::string getFolderName() const override { return "dummy"; }

  double getItem(size_t idx) {
    // idk some random equation to show that the value changes?
    return randomGetItem(idx);
  }
};

namespace dyno {

class DatasetProviderFixture : public ::testing::Test {
 public:
  DatasetProviderFixture() {}

 protected:
  virtual void SetUp() { fs::create_directory(sandbox); }
  virtual void TearDown() { fs::remove_all(sandbox); }

  const fs::path sandbox{"/tmp/sandbox"};
};

TEST_F(DatasetProviderFixture, testDataFolderStructureConstructor) {
  DoubleDataFolder::Ptr ddf = std::make_shared<DoubleDataFolder>();
  DataFolderStructure<double> ds("some_path", ddf);

  auto retrieved_ddf = ds.getDataFolder<0>();
  EXPECT_EQ(ddf, retrieved_ddf);
}

TEST_F(DatasetProviderFixture, testDataFolderStructureGetPath) {
  DoubleDataFolder::Ptr ddf = std::make_shared<DoubleDataFolder>();
  DataFolderStructure<double> ds("/tmp/some_path", ddf);

  const std::string folder_path = ds.getAbsoluteFolderPath<0>();
  const std::string expected_folder_path = "/tmp/some_path/dummy";
  EXPECT_EQ(folder_path, expected_folder_path);
}

TEST_F(DatasetProviderFixture, testGenericDatasetLoadingWithValidationFailure) {
  DoubleDataFolder::Ptr ddf = std::make_shared<DoubleDataFolder>();

  EXPECT_THROW({ GenericDataset<double>(sandbox, ddf); }, std::runtime_error);
}

TEST_F(DatasetProviderFixture, testGenericDatasetLoadingWithValidationSuccess) {
  DoubleDataFolder::Ptr ddf = std::make_shared<DoubleDataFolder>();

  const fs::path sandbox("/tmp/sandbox");

  std::ofstream{sandbox / ddf->getFolderName()};  // create regular file

  EXPECT_NO_THROW({ GenericDataset<double>(sandbox, ddf); });
}

TEST_F(DatasetProviderFixture, testGenericDatasetMockLoading) {
  DoubleDataFolder::Ptr ddf = std::make_shared<DoubleDataFolder>();

  std::ofstream{sandbox / ddf->getFolderName()};  // create regular file

  GenericDataset<double> gd(sandbox, ddf);

  std::vector<double>& data = gd.getDataVector<0>();
  EXPECT_EQ(data.size(), 0u);

  gd.load(1);
  EXPECT_EQ(data.size(), 1u);
  EXPECT_EQ(data.at(0), randomGetItem(0u));

  gd.load(3);
  EXPECT_EQ(data.size(), 3u);
  EXPECT_EQ(data.at(2), randomGetItem(2u));
}

// TEST_F(DatasetProviderFixture, testOptionalGenericdataset) {
//     FunctionalDataFolder<double>::Ptr loader1 =
//     std::make_shared<FunctionalDataFolder<double>>(
//         [=](size_t i) -> double {
//             return (double)i;
//         }
//     );

//     FunctionalDataFolder<int>::Ptr loader2 =
//     std::make_shared<FunctionalDataFolder<int>>(
//         [=](size_t i) ->int {
//             return i;
//         }
//     );

//     GenericDataset<double, std::optional<int>> dataset_with_optional(sandbox,
//     loader1, loader2)
// }

TEST(DynoDataset, testDummy) {
  // KittiDataLoader dd("/root/data/kitti/0000");
  // dd.spin();
}

// TEST(PlaybackGui, creation) {
//     PlaybackGui pg;

//     while(true) pg.draw();
// }

}  // namespace dyno
