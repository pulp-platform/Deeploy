#include "utils.h"

void *deeploy_malloc(const size_t size) { return snrt_l1alloc(size); }
