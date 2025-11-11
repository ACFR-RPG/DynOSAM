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

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dynosam/pipeline/ThreadSafeQueue.hpp"

void consumer(ThreadsafeQueue<std::string>& q,  // NOLINT
              const std::atomic_bool& kill_switch) {
  while (!kill_switch) {
    std::string msg = "No msg!";
    if (q.popBlocking(msg)) {
      VLOG(1) << "Got msg: " << msg << '\n';
    }
  }
  q.shutdown();
}

void producer_milliseconds(ThreadsafeQueue<std::string>& q,  // NOLINT
                           const std::atomic_bool& kill_switch, int delay) {
  while (!kill_switch) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    q.push("Hello World!");
  }
  q.shutdown();
}

void producer(ThreadsafeQueue<std::string>& q,  // NOLINT
              const std::atomic_bool& kill_switch) {
  while (!kill_switch) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    q.push("Hello World!");
  }
  q.shutdown();
}

void blockingProducer(ThreadsafeQueue<std::string>& q,  // NOLINT
                      const std::atomic_bool& kill_switch) {
  while (!kill_switch) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    q.pushBlockingIfFull("Hello World!", 5);
  }
  q.shutdown();
}

/* ************************************************************************* */
TEST(testThreadsafeQueue, popBlocking_by_reference) {
  ThreadsafeQueue<std::string> q;
  std::thread p([&] {
    q.push("Hello World!");
    q.push("Hello World 2!");
  });
  std::string s;
  q.popBlocking(s);
  EXPECT_EQ(s, "Hello World!");
  q.popBlocking(s);
  EXPECT_EQ(s, "Hello World 2!");
  q.shutdown();
  EXPECT_FALSE(q.popBlocking(s));
  EXPECT_EQ(s, "Hello World 2!");

  // Leave some time for p to finish its work.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(p.joinable());
  p.join();
}

/* ************************************************************************* */
TEST(testThreadsafeQueue, popBlocking_by_shared_ptr) {
  ThreadsafeQueue<std::string> q;
  std::thread p([&] {
    q.push("Hello World!");
    q.push("Hello World 2!");
  });
  std::shared_ptr<std::string> s = q.popBlocking();
  EXPECT_EQ(*s, "Hello World!");
  auto s2 = q.popBlocking();
  EXPECT_EQ(*s2, "Hello World 2!");
  q.shutdown();
  EXPECT_EQ(q.popBlocking(), nullptr);

  // Leave some time for p to finish its work.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(p.joinable());
  p.join();
}

/* ************************************************************************* */
TEST(testThreadsafeQueue, push) {
  ThreadsafeQueue<std::string> q;
  std::thread p([&] {
    q.push(std::string("Hello World!"));
    std::string s = "Hello World 2!";
    q.push(s);
  });
  std::shared_ptr<std::string> s = q.popBlocking();
  EXPECT_EQ(*s, "Hello World!");
  auto s2 = q.popBlocking();
  EXPECT_EQ(*s2, "Hello World 2!");
  q.shutdown();
  EXPECT_EQ(q.popBlocking(), nullptr);

  // Leave some time for p to finish its work.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(p.joinable());
  p.join();
}

TEST(testThreadsafeQueue, pointerContainer) {
  ThreadsafeQueue<std::shared_ptr<std::string>> q;
  std::thread p([&] {
    q.push(std::make_shared<std::string>("Hello World!"));
    auto s = std::make_shared<std::string>("Hello World 2!");
    q.push(s);
  });

  // wow this is gross...
  std::shared_ptr<std::shared_ptr<std::string>> s = q.popBlocking();
  EXPECT_EQ(**s, "Hello World!");
  auto s2 = q.popBlocking();
  EXPECT_EQ(**s2, "Hello World 2!");
  q.shutdown();
  EXPECT_EQ(q.popBlocking(), nullptr);

  // Leave some time for p to finish its work.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(p.joinable());
  p.join();
}

/* ************************************************************************* */
TEST(testThreadsafeQueue, pushBlockingIfFull) {
  // Here we test only its nominal push behavior, not the blocking behavior
  ThreadsafeQueue<std::string> q;
  std::thread p([&] {
    q.pushBlockingIfFull(std::string("Hello World!"), 2);
    std::string s = "Hello World 2!";
    q.pushBlockingIfFull(s, 2);
  });
  std::shared_ptr<std::string> s = q.popBlocking();
  EXPECT_EQ(*s, "Hello World!");
  auto s2 = q.popBlocking();
  EXPECT_EQ(*s2, "Hello World 2!");
  q.shutdown();
  EXPECT_EQ(q.popBlocking(), nullptr);

  // Leave some time for p to finish its work.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(p.joinable());
  p.join();
}

/* ************************************************************************* */
TEST(testThreadsafeQueue, producer_consumer) {
  ThreadsafeQueue<std::string> q;
  std::atomic_bool kill_switch(false);
  std::thread c(consumer, std::ref(q), std::ref(kill_switch));
  std::thread p(producer, std::ref(q), std::ref(kill_switch));

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Shutdown queue.\n";
  q.shutdown();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Resume queue.\n";
  q.resume();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Joining threads.\n";
  kill_switch = true;
  c.join();
  p.join();
  VLOG(1) << "Threads joined.\n";
}

/* ************************************************************************* */
TEST(testThreadsafeQueue, blocking_producer) {
  ThreadsafeQueue<std::string> q;
  std::atomic_bool kill_switch(false);
  std::thread p(blockingProducer, std::ref(q), std::ref(kill_switch));

  // Give plenty of time to the blockingProducer to fill-in completely the queue
  // and be blocked.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Shutdown queue.\n";
  q.shutdown();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Resume queue.\n";
  q.resume();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Joining thread.\n";
  q.shutdown();
  kill_switch = true;
  p.join();
  VLOG(1) << "Thread joined.\n";

  // Need to resume the queue to be able to pop...
  q.resume();

  // Expect non-empty queue.
  EXPECT_TRUE(!q.empty());
  size_t queue_size = 0;
  while (!q.empty()) {
    std::string output;
    EXPECT_TRUE(q.pop(output));
    EXPECT_EQ(output, "Hello World!");
    ++queue_size;
  }
  // Expect the size of the queue to be the maximum size of the queue
  EXPECT_EQ(queue_size, 5u);
}

/* ************************************************************************* */
TEST(testThreadsafeQueue, stress_test) {
  ThreadsafeQueue<std::string> q;
  std::atomic_bool kill_switch(false);
  std::vector<std::thread> cs;
  for (size_t i = 0; i < 10; i++) {
    // Create 10 consumers.
    cs.push_back(std::thread(consumer, std::ref(q), std::ref(kill_switch)));
  }
  std::vector<std::thread> ps;
  for (size_t i = 0; i < 10; i++) {
    // Create 10 producers.
    ps.push_back(std::thread(producer, std::ref(q), std::ref(kill_switch)));
  }
  std::vector<std::thread> blocking_ps;
  for (size_t i = 0; i < 10; i++) {
    // Create 10 producers.
    ps.push_back(
        std::thread(blockingProducer, std::ref(q), std::ref(kill_switch)));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Shutdown queue.\n";
  q.shutdown();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Resume queue.\n";
  q.resume();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  VLOG(1) << "Joining threads.\n";
  kill_switch = true;
  for (size_t i = 0; i < cs.size(); i++) {
    cs[i].join();
  }
  for (size_t i = 0; i < ps.size(); i++) {
    ps[i].join();
  }
  for (size_t i = 0; i < blocking_ps.size(); i++) {
    blocking_ps[i].join();
  }
  VLOG(1) << "Threads joined.\n";
}

TEST(testThreadsafeQueue, limited_queue_size) {
  ThreadsafeQueue<std::string> q(5);
  std::atomic_bool kill_switch(false);
  std::thread p(producer_milliseconds, std::ref(q), std::ref(kill_switch), 10);

  // Give plenty of time to the producer_milliseconds to fill-in completely the
  // queue and be blocked.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  // Expect the size of the queue to be the maximum size of the queue
  EXPECT_EQ(q.size(), 5u);

  q.shutdown();
  kill_switch = true;
  p.join();
}
