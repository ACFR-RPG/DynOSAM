/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */
#pragma once

#include <glog/logging.h>

#include <atomic>
#include <chrono>  // std::chrono::
#include <exception>
#include <functional>  // for function
#include <memory>
#include <string>
#include <thread>   // std::this_thread::sleep_for
#include <utility>  // for move
#include <vector>

#include "dynosam/pipeline/ThreadSafeQueue.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {

class PipelineBase {
 public:
  DYNO_POINTER_TYPEDEFS(PipelineBase)

  struct ReturnCode {
    enum Code {
      SUCCESS,
      IS_SHUTDOWN,
      OUTPUT_PUSH_FAILURE,
      PROCESSING_FAILURE,
      GET_PACKET_FAILURE
    };

    Code code_;

    ReturnCode() {}
    ReturnCode(Code code) : code_(code) {}
    explicit ReturnCode(int val) : code_(static_cast<Code>(val)) {}

    ReturnCode& operator=(Code code) {
      this->code_ = code;
      return *this;
    }
    operator Code() const { return code_; }
    operator bool() const { return code_ == Code::SUCCESS; }
  };

  using OnPipelineFailureCallback = std::function<void(ReturnCode)>;

  PipelineBase(const std::string& module_name) : module_name_(module_name) {}

  virtual ~PipelineBase() = default;
  /**
   * @brief Spins the pipeline via continuous calls to spinOnce until the
   * pipeline is shutdown.
   *
   * Blocking call
   *
   * @return true
   * @return false
   */
  bool spin();

  /**
   * @brief Returns true if the pipeline is currently processing data or if
   * these is still data to process. Checks is_thread_working_ and the
   * overwritten hasWork() function
   *
   * @return true
   * @return false
   */
  bool isWorking() const;

  /**
   * @brief Returns true if the pipeline has been shutdown by shutdown().
   *
   * @return true
   * @return false
   */
  inline bool isShutdown() const { return is_shutdown_; }

  /**
   * @brief Returns the given module name (for identifcation and debug printing)
   *
   * @return std::string
   */
  inline std::string getModuleName() const { return module_name_; }

  virtual void shutdown();

  /**
   * @brief Registers a callback to be triggered if spinOnce returns an invalid
   * return code
   *
   * @param callback_ const OnPipelineFailureCallback&
   */
  void registerOnFailureCallback(const OnPipelineFailureCallback& callback_);

  /**
   * @brief Virtual function that should shutdown all associated queues.
   * Generally this should shutdown input queues
   *
   */
  virtual void shutdownQueues() = 0;

  /**
   * @brief Virtual function that should check if (any of the) input queues have
   * work to do
   *
   * @return true
   * @return false
   */
  virtual bool hasWork() const = 0;

  /**
   * @brief Processes for one iteration of the pipeline.
   *
   * Pulls data from registetered input queues and sends the data to some output
   * queues. Called by spin().
   *
   * @return PipelineReturnStatus
   */
  virtual ReturnCode spinOnce() = 0;

 protected:
  const std::string module_name_;
  std::atomic_bool is_thread_working_{false};

 private:
  void notifyFailures(const ReturnCode& result);

 private:
  std::vector<OnPipelineFailureCallback> on_failure_callbacks_;
  //! Thread related members.
  std::atomic_bool is_shutdown_{false};
};

/**
 * @brief Defines type alias's for a single type (e.g INPUT or OUTPUT)
 * which will be used within a PipelineModule.
 *
 * Defines the type, a shared_ptr to a const type and a ThreadsafeQueue using
 * the const ptr type
 *
 * @tparam TYPE
 */
template <typename TYPE>
struct PipelineTypeTraits {
  using Type = TYPE;
  using TypeConstSharedPtr = std::shared_ptr<const Type>;
  using TypeQueue = ThreadsafeQueue<TypeConstSharedPtr>;
};

/**
 * @brief A PipelineModule is responsible for taking data from an input (or
 * input queues) processing it, and sending it to an output queue (or queues)
 *
 * The PipelineModule has several virtual functions used for processing which
 * are called when spinOnce() is called which should process one iteration of
 * the input data The call order is: input = getInputPacket() output =
 * process(input) pushOutputPacket(output)
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
class PipelineModule : public PipelineBase {
 public:
  using InputTypeTraits = PipelineTypeTraits<INPUT>;
  using OutputTypeTraits = PipelineTypeTraits<OUTPUT>;

  using InputConstSharedPtr = typename InputTypeTraits::TypeConstSharedPtr;
  //! The output is instead a shared ptr, since many users might need the output
  using OutputConstSharedPtr = typename OutputTypeTraits::TypeConstSharedPtr;

  using InputQueue = typename InputTypeTraits::TypeQueue;
  using OutputQueue = typename OutputTypeTraits::TypeQueue;

  using OnProcessCallback = std::function<void(const InputConstSharedPtr&,
                                               const OutputConstSharedPtr&)>;

  PipelineModule(const std::string& module_name) : PipelineBase(module_name) {}

  virtual ~PipelineModule() = default;

  /**
   * @brief Spin the pipeline once
   *
   * @return PipelineBase::ReturnCode
   */
  PipelineBase::ReturnCode spinOnce() override;

  /**
   * @brief Registers a function that will be called with the input and output
   * data packets generated from a spinOnce() call.
   *
   * These callbacks happen in the same thread as the module they are attached
   * to so should happen quickly
   *
   * @param callback const OnProcessCallback&
   */
  void registerOnProcessCallback(const OnProcessCallback& callback);

 protected:
  virtual InputConstSharedPtr getInputPacket() = 0;
  virtual OutputConstSharedPtr process(const InputConstSharedPtr& input) = 0;
  virtual bool pushOutputPacket(
      const OutputConstSharedPtr& output_packet) const = 0;

 private:
  // could be const?
  void emitProcessCallbacks(const InputConstSharedPtr& input_packet,
                            const OutputConstSharedPtr& output_packet);

 private:
  std::vector<OnProcessCallback> on_process_callbacks_;
};

template <typename PAYLOAD>
struct QueueRegistra {
 public:
  using This = QueueRegistra<PAYLOAD>;
  using Queue = typename PipelineTypeTraits<PAYLOAD>::TypeQueue;

  DYNO_POINTER_TYPEDEFS(This)

  QueueRegistra(std::vector<Queue*>* queues) : queues_(CHECK_NOTNULL(queues)) {}

  void registerQueue(Queue* queue) {
    CHECK_NOTNULL(queue);
    CHECK_NOTNULL(queues_);
    // TODO: not thread safe?
    queues_->push_back(queue);
  }

 private:
  //! Poitner to a vector of queue pointers (owned by the PipelineModule)
  //! Will only have the lifetime of the derived MIMOPipelineModule that created
  //! it
  std::vector<Queue*>* queues_;
};

// MultiInputMultiOutput -> abstract class and user must implement the
// getInputPacket
template <typename INPUT, typename OUTPUT>
class MIMOPipelineModule : public PipelineModule<INPUT, OUTPUT> {
 public:
  using Base = PipelineModule<INPUT, OUTPUT>;  // Base
  using This = MIMOPipelineModule<INPUT, OUTPUT>;

  DYNO_POINTER_TYPEDEFS(This)

  using typename Base::InputConstSharedPtr;
  using typename Base::InputQueue;
  using typename Base::OutputConstSharedPtr;
  using typename Base::OutputQueue;

  using OutputRegistra = QueueRegistra<OUTPUT>;

  //! Callback used to send data to other pipeline modules, makes use of
  //! shared pointer since the data may be shared between several modules.
  using OutputCallback =
      std::function<void(const OutputConstSharedPtr& output)>;

  MIMOPipelineModule(const std::string& module_name) : Base(module_name) {}

  void registerOutputQueue(OutputQueue* output_queue);

  /**
   * @brief Constructs an output registra for the internal vector of output
   * queues Basically provides a way to add queues to the pipeline separate to
   * the pipeline itself. This is useful as the registra is only templated on
   * the OUTPUT so we can simply any external templating which only concerns
   * itself with output types
   *
   * @return typename QueueRegistra<OUTPUT>::Ptr
   */
  typename OutputRegistra::Ptr getOutputRegistra() {
    return std::make_shared<OutputRegistra>(&output_queues_);
  }

 protected:
  bool pushOutputPacket(
      const OutputConstSharedPtr& output_packet) const override;

 private:
  //! Output callbacks that will be called on each spinOnce if
  //! an output is present.
  std::vector<OutputQueue*> output_queues_;
};

/** @brief SIMOPipelineModule Single Input Multiple Output (SIMO) pipeline
 * module.
 * Receives Input packets via a threadsafe queue, and sends output packets
 * to a list of registered callbacks with a specific signature.
 * This is useful when there are multiple modules expecting results from this
 * module.
 */
template <typename INPUT, typename OUTPUT>
class SIMOPipelineModule : public MIMOPipelineModule<INPUT, OUTPUT> {
 public:
  using This = SIMOPipelineModule<INPUT, OUTPUT>;
  using Base = MIMOPipelineModule<INPUT, OUTPUT>;

  DYNO_POINTER_TYPEDEFS(This)

  using typename Base::InputConstSharedPtr;
  using typename Base::InputQueue;
  using typename Base::OutputConstSharedPtr;
  using typename Base::OutputQueue;
  using typename Base::OutputRegistra;

  using Base::isShutdown;

  SIMOPipelineModule(const std::string& module_name, InputQueue* input_queue_,
                     bool parallel_run = true)
      : MIMOPipelineModule<INPUT, OUTPUT>(module_name),
        input_queue(CHECK_NOTNULL(input_queue_)),
        parallel_run_(parallel_run) {}

  This& parallelRun(bool parallel_run) {
    parallel_run_ = parallel_run;
    return *this;
  }

 protected:
  InputConstSharedPtr getInputPacket() override;

  void shutdownQueues() override;

  //! Checks if the module has work to do (should check input queues are empty)
  bool hasWork() const override;

 private:
  InputQueue* input_queue;
  std::atomic_bool parallel_run_;
};

template <typename INPUT, typename OUTPUT>
class FunctionalSIMOPipelineModule : public SIMOPipelineModule<INPUT, OUTPUT> {
 public:
  using Base = SIMOPipelineModule<INPUT, OUTPUT>;
  using typename Base::InputConstSharedPtr;
  using typename Base::InputQueue;
  using typename Base::OutputConstSharedPtr;
  using typename Base::OutputQueue;

  using ProcessFunc =
      std::function<OutputConstSharedPtr(const InputConstSharedPtr&)>;

  /**
   * @brief
   *
   * When constructing ProcessFunc the argument and return types must be
   explicit
   * e.g.
   * using VarPipeline = FunctionalSIMOPipelineModule<int, NullPipelinePayload>;
   * VarPipeline p("var_module", &input_queue,
      [](const VarPipeline::InputConstSharedPtr& var_ptr) ->
   VarPipeline::OutputConstSharedPtr { return
   std::make_shared<NullPipelinePayload>;
      });

      when defining the lambda.
   *
   * @param module_name
   * @param input_queue
   * @param func
   * @param parallel_run
   */
  FunctionalSIMOPipelineModule(const std::string& module_name,
                               InputQueue* input_queue, const ProcessFunc& func,
                               bool parallel_run = true)
      : Base(module_name, input_queue, parallel_run), func_(func) {}

  OutputConstSharedPtr process(const InputConstSharedPtr& input) override {
    return func_(input);
  }

 private:
  ProcessFunc func_;
};

}  // namespace dyno

#include "dynosam/pipeline/PipelineBase-inl.hpp"
