#include "persistent_cache_host.h"

#include <errno.h>
#include <sys/fcntl.h>
#include <sys/param.h>

int rvllm_persistent_cache_get_path(int descriptor, char *buffer, size_t capacity) {
    if (buffer == NULL || capacity < MAXPATHLEN) {
        errno = EINVAL;
        return -1;
    }
    return fcntl(descriptor, F_GETPATH, buffer);
}

int rvllm_persistent_cache_set_protection_class(int descriptor, int protection_class) {
    return fcntl(descriptor, F_SETPROTECTIONCLASS, protection_class);
}

int rvllm_persistent_cache_get_protection_class(int descriptor) {
    return fcntl(descriptor, F_GETPROTECTIONCLASS);
}
