#ifndef RVLLM_APPLE_H
#define RVLLM_APPLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RVLLM_APPLE_ABI_VERSION 1u
#define RVLLM_APPLE_ABI_VERSION_V2 2u
#define RVLLM_APPLE_ABI_VERSION_V3 3u
#define RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES 32u
#define RVLLM_APPLE_MAX_PERSISTENT_CACHE_BYTES 4294967296ull
#define RVLLM_APPLE_MAX_CACHE_NAMESPACE_BYTES 128u
#define RVLLM_APPLE_ERROR_MESSAGE_CAPACITY 512u

typedef int32_t RvllmAppleStatus;
#define RVLLM_APPLE_OK 0
#define RVLLM_APPLE_INVALID_ARGUMENT 1
#define RVLLM_APPLE_BACKEND_UNAVAILABLE 2
#define RVLLM_APPLE_QUEUE_FULL 3
#define RVLLM_APPLE_CANCELLED 4
#define RVLLM_APPLE_BUFFER_TOO_SMALL 5
#define RVLLM_APPLE_END_OF_STREAM 6
#define RVLLM_APPLE_INTERNAL_ERROR 7

#define RVLLM_APPLE_EVENT_TOKEN 1u
#define RVLLM_APPLE_EVENT_FINISHED 2u

#define RVLLM_APPLE_BACKEND_AUTOMATIC 0u
#define RVLLM_APPLE_BACKEND_METAL_ONLY 1u
#define RVLLM_APPLE_BACKEND_CORE_ML_PREFERRED 2u
#define RVLLM_APPLE_BACKEND_CORE_ML_ONLY 3u

#define RVLLM_APPLE_CACHE_DISABLED 0u
#define RVLLM_APPLE_CACHE_MEMORY_ONLY 1u
#define RVLLM_APPLE_CACHE_PERSISTENT_ENCRYPTED 2u

#define RVLLM_APPLE_WORKLOAD_INTERACTIVE 0u
#define RVLLM_APPLE_WORKLOAD_BALANCED 1u
#define RVLLM_APPLE_WORKLOAD_THROUGHPUT 2u

#define RVLLM_APPLE_MEMORY_CONSERVATIVE 0u
#define RVLLM_APPLE_MEMORY_BALANCED 1u
#define RVLLM_APPLE_MEMORY_MAXIMUM_PERFORMANCE 2u

#define RVLLM_APPLE_MEMORY_PRESSURE_NORMAL 0u
#define RVLLM_APPLE_MEMORY_PRESSURE_WARNING 1u
#define RVLLM_APPLE_MEMORY_PRESSURE_CRITICAL 2u

typedef struct RvllmAppleEngine RvllmAppleEngine;
typedef struct RvllmAppleRequest RvllmAppleRequest;

typedef struct RvllmAppleError {
    RvllmAppleStatus code;
    char message[RVLLM_APPLE_ERROR_MESSAGE_CAPACITY];
} RvllmAppleError;

typedef struct RvllmAppleEngineConfig {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t backend_policy;
    uint32_t cache_policy;
    uint32_t workload_profile;
    uint32_t memory_profile;
    uint32_t maximum_concurrency;
    uint32_t ingress_queue_capacity;
    uint32_t event_queue_capacity;
    uint8_t persistent_cache_consent;
    uint8_t _reserved[7];
    uint64_t hot_cache_bytes;
    uint64_t warm_cache_bytes;
    uint64_t persistent_cache_bytes;
} RvllmAppleEngineConfig;

typedef struct RvllmAppleEngineConfigV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t backend_policy;
    uint32_t cache_policy;
    uint32_t workload_profile;
    uint32_t memory_profile;
    uint32_t maximum_concurrency;
    uint32_t ingress_queue_capacity;
    uint32_t event_queue_capacity;
    uint8_t persistent_cache_consent;
    uint8_t _reserved[7];
    uint64_t hot_cache_bytes;
    uint64_t warm_cache_bytes;
    uint64_t persistent_cache_bytes;
    const uint8_t *model_package_path;
    size_t model_package_path_length;
    const uint8_t *resource_bundle_path;
    size_t resource_bundle_path_length;
} RvllmAppleEngineConfigV2;

/*
 * ABI v3 preserves the complete v2 prefix and appends an opt-in encrypted
 * persistent-cache capability. The path, exact 32-byte key, and required
 * engine-level tenant namespace are copied synchronously by
 * rvllm_apple_engine_create_v3. They are ignored only when all
 * persistent-cache fields are disabled/zero.
 *
 * A direct C host is responsible for obtaining the key from a public
 * platform keystore (Keychain on Apple platforms), keeping the cache root
 * private, and applying and verifying iOS Data Protection before creation.
 * The Swift package performs those host obligations for Swift callers.
 */
typedef struct RvllmAppleEngineConfigV3 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t backend_policy;
    uint32_t cache_policy;
    uint32_t workload_profile;
    uint32_t memory_profile;
    uint32_t maximum_concurrency;
    uint32_t ingress_queue_capacity;
    uint32_t event_queue_capacity;
    uint8_t persistent_cache_consent;
    uint8_t _reserved[7];
    uint64_t hot_cache_bytes;
    uint64_t warm_cache_bytes;
    uint64_t persistent_cache_bytes;
    const uint8_t *model_package_path;
    size_t model_package_path_length;
    const uint8_t *resource_bundle_path;
    size_t resource_bundle_path_length;
    const uint8_t *persistent_cache_root;
    size_t persistent_cache_root_length;
    const uint8_t *persistent_cache_key;
    size_t persistent_cache_key_length;
    const uint8_t *cache_namespace;
    size_t cache_namespace_length;
} RvllmAppleEngineConfigV3;

typedef struct RvllmAppleGenerateRequest {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *prompt_tokens;
    size_t prompt_token_count;
    uint32_t max_output_tokens;
    uint8_t priority;
    uint8_t _reserved[3];
    uint32_t cache_policy;
} RvllmAppleGenerateRequest;

typedef struct RvllmAppleBackendReport {
    /* 1 Metal, 2 public Core ML, 3 private Metal-prefill/ANE-decode research.
       The public creation API does not enable the private research route. */
    uint32_t selected_backend;
    uint32_t cache_tier;
    uint32_t matched_cache_tokens;
    uint32_t saved_prefill_tokens;
    uint64_t queue_time_ns;
    uint32_t batch_size;
    uint32_t padding_tokens;
    uint64_t prefill_time_ns;
    uint64_t decode_time_ns;
    uint64_t resident_memory_bytes;
    uint32_t thermal_state;
    uint8_t had_fallback;
    uint8_t _reserved[7];
} RvllmAppleBackendReport;

typedef struct RvllmAppleTokenEvent {
    uint32_t kind;
    uint64_t request_id;
    uint32_t index;
    uint32_t token_id;
    uint32_t finish_reason;
    size_t text_length;
    RvllmAppleBackendReport report;
} RvllmAppleTokenEvent;

/*
 * Opaque handles have unique ownership. Destroy each non-null handle exactly
 * once. Engine methods may be called concurrently. Only one receive operation
 * may be active per request. Cancel may run from another thread; destroy must
 * wait until receive returns.
 */
/* Legacy base-ABI query retained for existing hosts. */
uint32_t rvllm_apple_abi_version(void);
/* Highest versioned engine-configuration ABI supported by this library. */
uint32_t rvllm_apple_latest_abi_version(void);
RvllmAppleStatus rvllm_apple_engine_config_init(RvllmAppleEngineConfig *out_config);
RvllmAppleStatus rvllm_apple_engine_config_v2_init(RvllmAppleEngineConfigV2 *out_config);
RvllmAppleStatus rvllm_apple_engine_config_v3_init(RvllmAppleEngineConfigV3 *out_config);
RvllmAppleStatus rvllm_apple_generate_request_init(RvllmAppleGenerateRequest *out_request);

RvllmAppleStatus rvllm_apple_engine_create(
    const RvllmAppleEngineConfig *config,
    RvllmAppleEngine **out_engine,
    RvllmAppleError *error);
void rvllm_apple_engine_destroy(RvllmAppleEngine *engine);
RvllmAppleStatus rvllm_apple_engine_create_v2(
    const RvllmAppleEngineConfigV2 *config,
    RvllmAppleEngine **out_engine,
    RvllmAppleError *error);
RvllmAppleStatus rvllm_apple_engine_create_v3(
    const RvllmAppleEngineConfigV3 *config,
    RvllmAppleEngine **out_engine,
    RvllmAppleError *error);

RvllmAppleStatus rvllm_apple_engine_submit(
    RvllmAppleEngine *engine,
    const RvllmAppleGenerateRequest *request,
    RvllmAppleRequest **out_request,
    RvllmAppleError *error);

/*
 * Blocks for the next event. token text is copied without a terminator.
 * If capacity is insufficient, returns BUFFER_TOO_SMALL, writes text_length,
 * and retains the event so the caller can resize and retry without token loss.
 */
RvllmAppleStatus rvllm_apple_request_recv(
    RvllmAppleRequest *request,
    RvllmAppleTokenEvent *out_event,
    char *text_buffer,
    size_t text_capacity,
    RvllmAppleError *error);
RvllmAppleStatus rvllm_apple_request_cancel(RvllmAppleRequest *request);
void rvllm_apple_request_destroy(RvllmAppleRequest *request);

RvllmAppleStatus rvllm_apple_engine_handle_memory_pressure(
    RvllmAppleEngine *engine,
    uint32_t level,
    RvllmAppleError *error);

#ifdef __cplusplus
}
#endif

#endif /* RVLLM_APPLE_H */
