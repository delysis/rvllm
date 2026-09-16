#include "rvllm_apple.h"

#include <string.h>

/*
 * The source-only Swift package intentionally has no shipping native backend.
 * These test-target-only definitions make host-boundary unit tests link, and
 * fail closed if a test accidentally attempts inference.
 */

RvllmAppleStatus rvllm_apple_engine_config_v2_init(RvllmAppleEngineConfigV2 *out_config) {
    if (out_config == NULL) {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    memset(out_config, 0, sizeof(*out_config));
    out_config->abi_version = RVLLM_APPLE_ABI_VERSION_V2;
    return RVLLM_APPLE_OK;
}

RvllmAppleStatus rvllm_apple_engine_config_v3_init(RvllmAppleEngineConfigV3 *out_config) {
    if (out_config == NULL) {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    memset(out_config, 0, sizeof(*out_config));
    out_config->abi_version = RVLLM_APPLE_ABI_VERSION_V3;
    return RVLLM_APPLE_OK;
}

RvllmAppleStatus rvllm_apple_generate_request_init(RvllmAppleGenerateRequest *out_request) {
    if (out_request == NULL) {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    memset(out_request, 0, sizeof(*out_request));
    return RVLLM_APPLE_OK;
}

RvllmAppleStatus rvllm_apple_engine_create_v2(
    const RvllmAppleEngineConfigV2 *config,
    RvllmAppleEngine **out_engine,
    RvllmAppleError *out_error
) {
    (void)config;
    (void)out_engine;
    (void)out_error;
    return RVLLM_APPLE_BACKEND_UNAVAILABLE;
}

RvllmAppleStatus rvllm_apple_engine_create_v3(
    const RvllmAppleEngineConfigV3 *config,
    RvllmAppleEngine **out_engine,
    RvllmAppleError *out_error
) {
    (void)config;
    (void)out_engine;
    (void)out_error;
    return RVLLM_APPLE_BACKEND_UNAVAILABLE;
}

void rvllm_apple_engine_destroy(RvllmAppleEngine *engine) {
    (void)engine;
}

RvllmAppleStatus rvllm_apple_engine_submit(
    RvllmAppleEngine *engine,
    const RvllmAppleGenerateRequest *request,
    RvllmAppleRequest **out_request,
    RvllmAppleError *out_error
) {
    (void)engine;
    (void)request;
    (void)out_request;
    (void)out_error;
    return RVLLM_APPLE_BACKEND_UNAVAILABLE;
}

RvllmAppleStatus rvllm_apple_engine_handle_memory_pressure(
    RvllmAppleEngine *engine,
    uint32_t level,
    RvllmAppleError *out_error
) {
    (void)engine;
    (void)level;
    (void)out_error;
    return RVLLM_APPLE_BACKEND_UNAVAILABLE;
}

RvllmAppleStatus rvllm_apple_request_recv(
    RvllmAppleRequest *request,
    RvllmAppleTokenEvent *out_event,
    char *text_buffer,
    size_t text_buffer_capacity,
    RvllmAppleError *out_error
) {
    (void)request;
    (void)out_event;
    (void)text_buffer;
    (void)text_buffer_capacity;
    (void)out_error;
    return RVLLM_APPLE_BACKEND_UNAVAILABLE;
}

RvllmAppleStatus rvllm_apple_request_cancel(RvllmAppleRequest *request) {
    (void)request;
    return RVLLM_APPLE_BACKEND_UNAVAILABLE;
}

void rvllm_apple_request_destroy(RvllmAppleRequest *request) {
    (void)request;
}
