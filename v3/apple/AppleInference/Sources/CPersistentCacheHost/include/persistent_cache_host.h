#ifndef RVLLM_PERSISTENT_CACHE_HOST_H
#define RVLLM_PERSISTENT_CACHE_HOST_H

#include <stddef.h>

int rvllm_persistent_cache_get_path(int descriptor, char *buffer, size_t capacity);
int rvllm_persistent_cache_set_protection_class(int descriptor, int protection_class);
int rvllm_persistent_cache_get_protection_class(int descriptor);

#endif
