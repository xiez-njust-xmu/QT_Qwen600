// inference_engine_impl.cu
// CUDA compilation unit — implements InferenceEngine via PIMPL.

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "config.h"
#include "sampler.h"

// construct_path is declared in tokenizer.h but defined here
void construct_path(char* out_path, size_t out_size, const char* dir, const char* filename) {
    size_t len = strlen(dir);
    if (len > 0 && (dir[len - 1] == '/' || dir[len - 1] == '\\')) {
        snprintf(out_path, out_size, "%s%s", dir, filename);
    } else {
        snprintf(out_path, out_size, "%s/%s", dir, filename);
    }
}

#include "tokenizer.h"
#include "qwen_model.cuh"
#include "inference_engine.h"

struct InferenceEngine::Impl {
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;
    bool initialized;
    int pos;
    bool is_first_turn;

    Impl() : initialized(false), pos(0), is_first_turn(true) {}
};

InferenceEngine::InferenceEngine() : impl_(new Impl()) {}

InferenceEngine::~InferenceEngine() {
    shutdown();
    delete impl_;
}

bool InferenceEngine::init(const std::string& model_dir, const InferenceParams& params) {
    if (impl_->initialized) shutdown();

    char model_path[1024];
    construct_path(model_path, sizeof(model_path), model_dir.c_str(), "model.safetensors");

    build_transformer(&impl_->transformer, model_path);
    build_tokenizer(&impl_->tokenizer, model_dir.c_str(), params.enable_thinking);
    build_sampler(&impl_->sampler, params.temperature, params.top_p, params.top_k,
                  (unsigned long long)time(NULL));

    impl_->pos = 0;
    impl_->is_first_turn = true;
    impl_->initialized = true;
    return true;
}

void InferenceEngine::shutdown() {
    if (!impl_->initialized) return;
    free_sampler(&impl_->sampler);
    free_tokenizer(&impl_->tokenizer);
    free_transformer(&impl_->transformer);
    impl_->initialized = false;
}

void InferenceEngine::update_sampler(float temperature, float top_p, int top_k) {
    impl_->sampler.temperature = temperature;
    impl_->sampler.topp = top_p;
    impl_->sampler.top_k = (top_k == 0) ? VOCAB_SIZE : top_k;
}

void InferenceEngine::reset_context() {
    impl_->pos = 0;
    impl_->is_first_turn = true;
}

void InferenceEngine::generate(const std::string& user_input,
                               const std::string& system_prompt,
                               std::function<void(const std::string&)> token_callback,
                               const volatile bool* stop_flag) {
    if (!impl_->initialized) return;

    if (impl_->pos >= SEQ_LEN) {
        reset_context();
        token_callback("[context cleared]\n");
    }

    char rendered_prompt[PROMPT_BUFFER_SIZE];
    if (impl_->is_first_turn && !system_prompt.empty()) {
        snprintf(rendered_prompt, PROMPT_BUFFER_SIZE,
                 impl_->tokenizer.system_prompt_template,
                 system_prompt.c_str(), user_input.c_str());
    } else {
        snprintf(rendered_prompt, PROMPT_BUFFER_SIZE,
                 impl_->tokenizer.prompt_template, user_input.c_str());
    }
    impl_->is_first_turn = false;

    int* prompt_tokens = (int*)malloc(PROMPT_BUFFER_SIZE * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&impl_->tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);

    int token = prompt_tokens[0];
    int next = 0;

    for (int local_pos = 0; local_pos < num_prompt_tokens + SEQ_LEN; local_pos++) {
        if (stop_flag && *stop_flag) break;
        if (impl_->pos >= SEQ_LEN) break;

        token = (local_pos < num_prompt_tokens) ? prompt_tokens[local_pos] : next;
        float* logits = forward(&impl_->transformer, token, impl_->pos);
        impl_->pos++;
        next = sample(&impl_->sampler, logits);

        if (local_pos >= num_prompt_tokens) {
            if (next == (int)impl_->tokenizer.eos_token_id) break;
            char* piece = decode(&impl_->tokenizer, next);
            if (piece) token_callback(std::string(piece));
        }
    }

    free(prompt_tokens);
}

bool InferenceEngine::is_initialized() const { return impl_->initialized; }
int InferenceEngine::current_pos() const { return impl_->pos; }
