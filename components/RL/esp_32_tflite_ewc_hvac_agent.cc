/*
Minimal ESP32 example: load two TFLite Micro models (actor.tflite and critic.tflite) from SPIFFS,
run inference periodically, compute a simple TD(0) advantage, compute per-weight gradients (mocked),
estimate Fisher (squared grads running average), and apply an EWC-style regularized update to
in-memory float weight vectors.

Important notes:
 - This example treats TFLite Micro interpreter as a black-box for forward inference only.
 - Updating TFLite model weights on-device is non-trivial because TFLite Micro stores operators
   and flatbuffer bytes; here we emulate trainable weights as separate float arrays ("shadow weights").
 - For real OTA weight updates use a custom binary container holding raw float weights, or use
   a framework that supports writable models.
 - This file is a minimal skeleton; glue it into your ESP-IDF project and adjust NN layer shapes.
*/

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_vfs_spiffs.h"
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>

// TFLite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

static const char* TAG = "hvac_ewc";
  
// ----------------- EWC memory containers -----------------
struct TrainableArray {
    std::vector<float> weights;      // current trainable weights (shadow)
    std::vector<float> fisher;       // estimated fisher diagonal
    std::vector<float> old_weights;  // snapshot at time of consolidation

    TrainableArray() {}
    TrainableArray(size_t n) { resize(n); }
    void resize(size_t n) {
        weights.assign(n, 0.0f);
        fisher.assign(n, 0.0f);
        old_weights.assign(n, 0.0f);
    }
};

// For this example we'll assume actor and critic each have some number of "trainable" floats
// In practice you would map these to your NN weight arrays or keep them in a separate dense head.

static TrainableArray actor_params;
static TrainableArray critic_params;

// Hyperparams
static const float LEARNING_RATE = 1e-3f;
static const float EWC_LAMBDA = 100.0f; // regularisation strength
static const float FISHER_DECAY = 0.99f; // running average for fisher

// Mock gradient computation: in a real setup you must compute true gradients w.r.t. weights
// Here we'll create a deterministic pseudo-gradient from inputs for demonstration
void compute_mock_gradients(const std::vector<float>& observation,
                            const std::vector<float>& action_probs,
                            float value,
                            std::vector<float>& out_grads_actor,
                            std::vector<float>& out_grads_critic)
{
    size_t na = actor_params.weights.size();
    size_t nc = critic_params.weights.size();
    out_grads_actor.assign(na, 0.0f);
    out_grads_critic.assign(nc, 0.0f);

    // Simple heuristic: correlate grads with obs elements
    for (size_t i = 0; i < na; ++i) {
        float s = 0.0f;
        for (size_t j = 0; j < observation.size(); ++j) s += observation[j] * (1.0f + (float)((i + j) % 3));
        out_grads_actor[i] = 0.001f * s * (0.5f - action_probs[i % action_probs.size()]);
    }
    for (size_t i = 0; i < nc; ++i) {
        float s = 0.0f;
        for (size_t j = 0; j < observation.size(); ++j) s += observation[j] * (1.0f + (float)((i + j) % 5));
        out_grads_critic[i] = 0.001f * s * (value - 0.5f);
    }
}

// Apply EWC update: gradient descent + EWC quadratic penalty using fisher diagonal
void apply_ewc_update(const std::vector<float>& grads_actor,
                      const std::vector<float>& grads_critic)
{
    // check sizes
    if (grads_actor.size() != actor_params.weights.size() ||
        grads_critic.size() != critic_params.weights.size()) {
        ESP_LOGE(TAG, "Gradient size mismatch");
        return;
    }

    // Update fisher running average (use squared grads as proxy)
    for (size_t i = 0; i < actor_params.fisher.size(); ++i) {
        float sq = grads_actor[i] * grads_actor[i];
        actor_params.fisher[i] = FISHER_DECAY * actor_params.fisher[i] + (1 - FISHER_DECAY) * sq;
    }
    for (size_t i = 0; i < critic_params.fisher.size(); ++i) {
        float sq = grads_critic[i] * grads_critic[i];
        critic_params.fisher[i] = FISHER_DECAY * critic_params.fisher[i] + (1 - FISHER_DECAY) * sq;
    }

    // Gradient descent with EWC penalty: grad + lambda * F * (w - w_old)
    for (size_t i = 0; i < actor_params.weights.size(); ++i) {
        float penalty = EWC_LAMBDA * actor_params.fisher[i] * (actor_params.weights[i] - actor_params.old_weights[i]);
        actor_params.weights[i] -= LEARNING_RATE * (grads_actor[i] + penalty);
    }
    for (size_t i = 0; i < critic_params.weights.size(); ++i) {
        float penalty = EWC_LAMBDA * critic_params.fisher[i] * (critic_params.weights[i] - critic_params.old_weights[i]);
        critic_params.weights[i] -= LEARNING_RATE * (grads_critic[i] + penalty);
    }
}

// Snapshot consolidation (call this after a training phase to freeze current params)
void consolidate_params()
{
    actor_params.old_weights = actor_params.weights;
    critic_params.old_weights = critic_params.weights;
}

// ----------------- TFLite Micro setup -----------------
// Arena size (tune according to models)
static const int kTensorArenaSize = 32 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

struct TFLiteModelContext {
    const uint8_t* model_buf;
    size_t model_size;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    tflite::MicroErrorReporter micro_error;
    tflite::MicroMutableOpResolver<10> resolver; // allow up to 10 ops
    bool valid;

    TFLiteModelContext(): model_buf(nullptr), model_size(0), model(nullptr), interpreter(nullptr), input_tensor(nullptr), output_tensor(nullptr), valid(false) {}
};

// We'll keep two contexts: actor and critic interpreter
static TFLiteModelContext actor_ctx;
static TFLiteModelContext critic_ctx;

bool setup_tflite_model(TFLiteModelContext& ctx, const char* path)
{
    size_t sz = 0;
    uint8_t* buf = read_file_to_buffer(path, &sz);
    if (!buf) return false;

    ctx.model_buf = buf;
    ctx.model_size = sz;
    ctx.model = tflite::GetModel(ctx.model_buf);
    if (ctx.model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "TFLite model version mismatch: %d", ctx.model->version());
        free((void*)ctx.model_buf);
        return false;
    }

    // Register a minimal set of ops used by your models. Add more if needed.
    ctx.resolver.AddFullyConnected();
    ctx.resolver.AddSoftmax();
    ctx.resolver.AddReshape();
    ctx.resolver.AddRelu();
    ctx.resolver.AddQuantize();
    ctx.resolver.AddDequantize();

    ctx.interpreter = new tflite::MicroInterpreter(ctx.model, ctx.resolver, tensor_arena, kTensorArenaSize, &ctx.micro_error);
    TfLiteStatus alloc_status = ctx.interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        delete ctx.interpreter;
        free((void*)ctx.model_buf);
        return false;
    }
    ctx.input_tensor = ctx.interpreter->input(0);
    ctx.output_tensor = ctx.interpreter->output(0);
    ctx.valid = true;
    ESP_LOGI(TAG, "Loaded TFLite model %s, input dims: %d", path, ctx.input_tensor->bytes);
    return true;
}

// Helper to run inference given float vector input and get float vector output
bool run_tflite_inference(TFLiteModelContext& ctx, const std::vector<float>& input, std::vector<float>& output)
{
    if (!ctx.valid) return false;
    TfLiteTensor* in = ctx.input_tensor;
    // assume float input
    if (in->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Model input is not float32");
        return false;
    }
    size_t n = 1;
    for (int i = 0; i < in->dims->size; ++i) n *= in->dims->data[i];
    if (input.size() != n) {
        ESP_LOGE(TAG, "Input size mismatch: got %d need %d", (int)input.size(), (int)n);
        return false;
    }

    float* data_ptr = ctx.interpreter->typed_input_tensor<float>(0);
    memcpy(data_ptr, input.data(), n * sizeof(float));

    TfLiteStatus status = ctx.interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return false;
    }

    TfLiteTensor* out = ctx.output_tensor;
    if (out->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Output is not float32");
        return false;
    }
    size_t m = 1;
    for (int i = 0; i < out->dims->size; ++i) m *= out->dims->data[i];
    output.resize(m);
    memcpy(output.data(), ctx.interpreter->typed_output_tensor<float>(0), m * sizeof(float));
    return true;
}

// ----------------- HVAC agent task -----------------
extern "C" void hvac_agent(void)
{
    vTaskDelay(pdMS_TO_TICKS(3000)); // wait for wifi/spiffs

    if (!spiffs_init()) {
        ESP_LOGE(TAG, "SPIFFS init failed");
        vTaskDelete(NULL);
        return;
    }

    // initialize tiny trainable arrays sizes (example sizes)
    actor_params.resize(12);   // e.g. 12 float params
    critic_params.resize(8);   // e.g. 8 float params
    // optionally initialize with small random values
    for (size_t i=0;i<actor_params.weights.size();++i) actor_params.weights[i] = 0.01f * (float)(i+1);
    for (size_t i=0;i<critic_params.weights.size();++i) critic_params.weights[i] = 0.01f * (float)(i+1);

    // load TFLite models from SPIFFS
    if (!setup_tflite_model(actor_ctx, "/spiffs1/actor.tflite")) {
        ESP_LOGE(TAG, "Load actor model failed");
    }
    if (!setup_tflite_model(critic_ctx, "/spiffs1/critic.tflite")) {
        ESP_LOGE(TAG, "Load critic model failed");
    }

    // main loop
    while (true) {
        // build observation (read your sensors / PID state)
        std::vector<float> observation(5, 0.0f);
        // TODO: replace with actual sensors
        observation[0] = 25.0f; // temp
        observation[1] = 60.0f; // humidity
        observation[2] = 300.0f; // light
        observation[3] = 0.1f; // co2
        observation[4] = 0.0f; // placeholder

        // run actor & critic TFLite inference
        std::vector<float> actor_out;
        std::vector<float> critic_out;
        bool ok_actor = run_tflite_inference(actor_ctx, observation, actor_out);
        bool ok_critic = run_tflite_inference(critic_ctx, observation, critic_out);

        if (!ok_actor || !ok_critic) {
            ESP_LOGW(TAG, "Inference failed, skipping step");
            vTaskDelay(pdMS_TO_TICKS(5000));
            continue;
        }

        // assume actor_out is action probabilities and critic_out is scalar value
        float value = critic_out.size() > 0 ? critic_out[0] : 0.0f;

        // normalize actor_out to sum=1 (softmax output might already be normalized)
        float sum = 0.0f;
        for (float v: actor_out) sum += v;
        if (sum > 1e-6f) for (auto &v: actor_out) v /= sum;

        ESP_LOGI(TAG, "Action probs:");
        for (float v: actor_out) printf("%.3f ", v);
        printf("\n");
        ESP_LOGI(TAG, "Value: %.4f", value);

        // compute TD(0) style advantage with a fake reward & gamma
        float reward = 0.0f; // TODO: compute from env
        float gamma = 0.99f;
        // For demo, treat next value as 0 (no bootstrap), so advantage = reward - value
        float advantage_scalar = reward - value;

        // compute mock gradients (replace with real gradients if you have them)
        std::vector<float> grads_actor;
        std::vector<float> grads_critic;
        compute_mock_gradients(observation, actor_out, value, grads_actor, grads_critic);

        // apply EWC update
        apply_ewc_update(grads_actor, grads_critic);

        // print some weights and fisher
        ESP_LOGI(TAG, "Actor weights[0..3]: %.6f %.6f %.6f %.6f", actor_params.weights[0], actor_params.weights[1], actor_params.weights[2], actor_params.weights[3]);
        ESP_LOGI(TAG, "Actor fisher[0..3]: %.6f %.6f %.6f %.6f", actor_params.fisher[0], actor_params.fisher[1], actor_params.fisher[2], actor_params.fisher[3]);

        // update old action probs if needed (kept by user code)

        // optional: after some steps, consolidate
        static int step_count = 0;
        step_count++;
        if (step_count % 100 == 0) {
            consolidate_params();
            ESP_LOGI(TAG, "Consolidated parameters (snapshot taken)");
        }

        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
