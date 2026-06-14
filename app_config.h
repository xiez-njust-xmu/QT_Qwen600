#pragma once

// Path to model directory containing:
//   model.safetensors, tokenizer.bin, template_*.txt
#define MODEL_DIR "Qwen3-0.6B"

// Default generation parameters
#define DEFAULT_TEMPERATURE 0.7f
#define DEFAULT_TOP_P 0.8f
#define DEFAULT_TOP_K 20
#define DEFAULT_SYSTEM_PROMPT "You are a helpful assistant."
//#define DEFAULT_SYSTEM_PROMPT "You are a professional translator. Translate the user's input into English. Only output the translation, no explanations."
#define DEFAULT_ENABLE_THINKING 0

//对于思考模式，使用 Temperature=0.6，TopP=0.95，TopK=20
//对于思考模式，使用 Temperature=0.7，TopP=0.8，TopK=20
