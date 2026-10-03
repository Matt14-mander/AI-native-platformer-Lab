#pragma once
#include <stddef.h>

#ifdef _WIN32
#define PTI_API __declspec(dllexport)
#else
#define PTI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif
PTI_API int pti_abi_version(void);
/* Last error on the calling thread, valid until its next bridge call. */
PTI_API const char* pti_last_error(void);
/* Each handle owns a prepared model/context. Returns NULL on error. */
PTI_API void* pti_create(const char* path, const char* input_name,
                         const char* output_name, size_t inputs, size_t outputs,
                         int fuse_relu);
/* Input/output are contiguous FP32 buffers; counts must match creation. */
PTI_API int pti_infer(void* handle, const float* input, size_t inputs,
                     float* output, size_t outputs);
/* Serialize destruction with inference; each live handle is destroyed once. */
PTI_API void pti_destroy(void* handle);
#ifdef __cplusplus
}
#endif
