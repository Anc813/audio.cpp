#pragma once

#include <initializer_list>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace engine::io {

#ifdef _WIN32
using DynamicLibraryHandle = HMODULE;
#else
using DynamicLibraryHandle = void *;
#endif

inline DynamicLibraryHandle open_dynamic_library(std::initializer_list<const char *> names) {
    for (const char * name : names) {
#ifdef _WIN32
        DynamicLibraryHandle handle = LoadLibraryA(name);
#else
        DynamicLibraryHandle handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
        if (handle != nullptr) {
            return handle;
        }
    }
    return nullptr;
}

inline void close_dynamic_library(DynamicLibraryHandle handle) {
    if (handle == nullptr) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

inline void * dynamic_library_symbol(DynamicLibraryHandle handle, const char * name) {
    if (handle == nullptr) {
        return nullptr;
    }
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(handle, name));
#else
    return dlsym(handle, name);
#endif
}

}  // namespace engine::io
