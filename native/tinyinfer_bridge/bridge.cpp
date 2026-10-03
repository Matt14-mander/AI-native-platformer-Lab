#include "bridge.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "tinyinfer/tinyinfer.h"

namespace {
thread_local char last_error[2048] = {};
void error(const char* message) noexcept {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
}

tinyinfer::Model load_model(const char* path, bool fuse_relu) {
    auto model = tinyinfer::onnx::OnnxImporter{}.load(path);
    if (!fuse_relu) return model;
    tinyinfer::optimizer::PassManager passes;
    passes.add_pass(std::make_unique<tinyinfer::optimizer::GemmActivationFusionPass>());
    return passes.run(model).model;
}

struct Session {
    tinyinfer::CpuExecutionPlan plan;
    tinyinfer::CpuExecutionContext context;
    std::string input_name;
    std::string output_name;
    size_t inputs;
    size_t outputs;
    std::mutex mutex;

    Session(const char* path, const char* input, const char* output,
            size_t input_count, size_t output_count, bool fuse_relu)
        : plan(checked_model(path, input, output, input_count, output_count, fuse_relu)),
          context(plan.create_context()), input_name(input), output_name(output),
          inputs(input_count), outputs(output_count) {}

    static tinyinfer::Model checked_model(const char* path, const char* input,
            const char* output, size_t inputs, size_t outputs, bool fuse_relu) {
        auto model = load_model(path, fuse_relu);
        if (model.inputs().size() != 1 || model.outputs().size() != 1)
            throw std::invalid_argument("actor must have one input and one output");
        const auto& in = model.graph().value(model.input_id(input)).spec;
        const auto& out = model.graph().value(model.output_id(output)).spec;
        if (in.dtype != tinyinfer::DataType::Float32 ||
            out.dtype != tinyinfer::DataType::Float32 ||
            in.shape != tinyinfer::Shape{1, static_cast<int64_t>(inputs)} ||
            out.shape != tinyinfer::Shape{1, static_cast<int64_t>(outputs)})
            throw std::invalid_argument("actor requires static [1,N] -> [1,M] FP32 tensors");
        return model;
    }
};
}  // namespace

extern "C" {
int pti_abi_version(void) { return 1; }
const char* pti_last_error(void) { return last_error; }

void* pti_create(const char* path, const char* input_name, const char* output_name,
                 size_t inputs, size_t outputs, int fuse_relu) {
    error("");
    try {
        if (!path || !input_name || !output_name || !inputs || !outputs)
            throw std::invalid_argument("null names/path or zero tensor dimensions");
        if (inputs > 4096 || outputs > 4096 || (fuse_relu != 0 && fuse_relu != 1))
            throw std::invalid_argument("unsupported actor dimensions or fusion flag");
        return new Session(path, input_name, output_name, inputs, outputs, fuse_relu != 0);
    } catch (const std::exception& e) { error(e.what()); }
      catch (...) { error("unknown TinyInfer creation error"); }
    return nullptr;
}

int pti_infer(void* handle, const float* input, size_t inputs, float* output, size_t outputs) {
    error("");
    try {
        if (!handle || !input || !output) throw std::invalid_argument("null inference argument");
        auto& session = *static_cast<Session*>(handle);
        std::lock_guard<std::mutex> guard(session.mutex);
        if (inputs != session.inputs || outputs != session.outputs)
            throw std::invalid_argument("inference buffer counts do not match model");
        tinyinfer::Tensor tensor({1, static_cast<int64_t>(inputs)});
        std::copy(input, input + inputs, tensor.data_f32());
        session.context.bind_input(session.input_name, std::move(tensor));
        session.context.run();
        auto result = session.context.output(session.output_name);
        if (result.dtype() != tinyinfer::DataType::Float32 ||
            result.numel() != outputs || !result.is_contiguous())
            throw std::runtime_error("unexpected TinyInfer output layout");
        std::copy(result.data_f32(), result.data_f32() + outputs, output);
        return 0;
    } catch (const std::exception& e) { error(e.what()); }
      catch (...) { error("unknown TinyInfer inference error"); }
    return -1;
}

void pti_destroy(void* handle) {
    delete static_cast<Session*>(handle);
}
}  // extern "C"
