#pragma once

#include <string>
#include <functional>

// Pure C++ header — no CUDA includes.
// Implementation lives in inference_engine_impl.cu.

struct InferenceParams {
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 20;
    int enable_thinking = 0;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    bool init(const std::string& model_dir, const InferenceParams& params);
    void shutdown();
    void update_sampler(float temperature, float top_p, int top_k);
    void reset_context();

    void generate(const std::string& user_input,
                  const std::string& system_prompt,
                  std::function<void(const std::string&)> token_callback,
                  const volatile bool* stop_flag = nullptr);

    bool is_initialized() const;
    int current_pos() const;

private:
    struct Impl;
    Impl* impl_;
};
