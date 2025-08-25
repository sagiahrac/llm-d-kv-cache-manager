/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <unistd.h> // for getpid() and usleep()

#include "cgo_functions.h"

// Global variables for caching
PyObject* g_chat_template_module = NULL;
PyObject* g_render_jinja_template_func = NULL;
PyObject* g_get_model_chat_template_func = NULL;
int g_initialized = 0;
int g_python_initialized = 0;

// Process-level global initialization tracking
static int g_process_initialized = 0;
static int g_finalized = 0; 
static pid_t g_init_pid = 0;

// Thread safety for initialization
static PyThread_type_lock g_init_lock = NULL;
static PyThread_type_lock g_python_init_lock = NULL;

// === ORIGINAL FUNCTION IMPLEMENTATIONS ===

// Initialize Python interpreter
int Py_InitializeGo() {
    // Process-level initialization check
    if (g_process_initialized) {
        if (g_init_pid != getpid()) {
            printf("[C] Py_InitializeGo WARNING - Different PID trying to initialize (init_pid: %d, current_pid: %d)\n", g_init_pid, getpid());
        } else {
            printf("[C] Py_InitializeGo - Already initialized in this process (PID: %d)\n", getpid());
        }
        return 0;
    }

    // Thread-safe initialization check
    if (g_python_init_lock == NULL) {
        g_python_init_lock = PyThread_allocate_lock();
        if (g_python_init_lock == NULL) {
            printf("[C] Py_InitializeGo ERROR - Failed to allocate Python init lock\n");
            return -1;
        }
    }

    PyThread_acquire_lock(g_python_init_lock, NOWAIT_LOCK);

    // Double-check after acquiring lock
    if (g_python_initialized) {
        printf("[C] Py_InitializeGo - Python already initialized globally\n");
        PyThread_release_lock(g_python_init_lock);
        return 0;
    }

    if (!Py_IsInitialized()) {
        // Initialize Python interpreter
        Py_Initialize();

        // Initialize threading support BEFORE any other operations
        PyEval_InitThreads();

        // Release the GIL so other threads can acquire it
        PyEval_ReleaseThread(PyThreadState_Get());
    }

    g_python_initialized = 1;
    g_process_initialized = 1;
    g_init_pid = getpid();
    PyThread_release_lock(g_python_init_lock);

    return 0;
}

// Finalize Python interpreter
void Py_FinalizeGo() {
    // Prevent multiple finalizations
    if (g_finalized) {
        printf("[C] Py_FinalizeGo - Already finalized, skipping\n");
        return;
    }
    
    // Mark as finalized first to prevent race conditions
    g_finalized = 1;
    
    // Clean up module references safely
    if (g_render_jinja_template_func) {
        Py_DECREF(g_render_jinja_template_func);
        g_render_jinja_template_func = NULL;
    }
    if (g_get_model_chat_template_func) {
        Py_DECREF(g_get_model_chat_template_func);
        g_get_model_chat_template_func = NULL;
    }
    if (g_chat_template_module) {
        Py_DECREF(g_chat_template_module);
        g_chat_template_module = NULL;
    }
    
    // Reset state without finalizing Python
    // Python will be cleaned up when the process exits
    g_python_initialized = 0;
    g_process_initialized = 0;
    g_initialized = 0;
}

// CGo cannot call C macros, so we wrap PyRun_SimpleString in a function
int Go_PyRun_SimpleString(const char* code) {
    return PyRun_SimpleString(code);
}

// Wrapper for PyImport_AddModule
PyObject* Go_PyImport_AddModule(const char* name) {
    return PyImport_AddModule(name);
}

// Wrapper for PyModule_GetDict
PyObject* Go_PyModule_GetDict(PyObject* module) {
    return PyModule_GetDict(module);
}

// Wrapper for PyDict_GetItemString
PyObject* Go_PyDict_GetItemString(PyObject* dict, const char* key) {
    return PyDict_GetItemString(dict, key);
}

// Helper function to convert Python string to Go string
const char* PyUnicode_AsGoString(PyObject* obj) {
    return PyUnicode_AsUTF8(obj);
}

// === NEW CACHING FUNCTION IMPLEMENTATIONS ===

// Initialize the cached module and functions (call once at startup)
int Py_InitChatTemplateModule() {
    
    // Thread-safe initialization check
    if (g_init_lock == NULL) {
        g_init_lock = PyThread_allocate_lock();
        if (g_init_lock == NULL) {
            printf("[C] Py_InitChatTemplateModule ERROR - Failed to allocate module init lock\n");
            return -1;
        }
    }
    
    PyThread_acquire_lock(g_init_lock, NOWAIT_LOCK);
    
    // Check if already initialized
    if (g_initialized) {
        printf("[C] Py_InitChatTemplateModule - Already initialized globally, returning\n");
        PyThread_release_lock(g_init_lock);
        return 0;
    }
    
    // Ensure Python is initialized
    if (!g_python_initialized) {
        printf("[C] Py_InitChatTemplateModule ERROR - Python not initialized\n");
        PyThread_release_lock(g_init_lock);
        return -1;
    }
    
    // Acquire GIL for module initialization
    PyGILState_STATE gil_state = PyGILState_Ensure();
    

    
    // Import the chat template wrapper module AFTER setting up the path
    g_chat_template_module = PyImport_ImportModule("render_jinja_template_wrapper");
    if (!g_chat_template_module) {
        printf("[C] Py_InitChatTemplateModule ERROR - Failed to import render_jinja_template_wrapper module\n");
        PyErr_Print();
        PyGILState_Release(gil_state);
        PyThread_release_lock(g_init_lock);
        return -1;
    }
    
    // Get the module dictionary
    PyObject* module_dict = PyModule_GetDict(g_chat_template_module);
    if (!module_dict) {
        printf("[C] Py_InitChatTemplateModule ERROR - Failed to get module dictionary\n");
        PyGILState_Release(gil_state);
        PyThread_release_lock(g_init_lock);
        return -1;
    }
    
    // Get the render_jinja_template function
    g_render_jinja_template_func = PyDict_GetItemString(module_dict, "render_jinja_template");
    if (!g_render_jinja_template_func || !PyCallable_Check(g_render_jinja_template_func)) {
        printf("[C] Py_InitChatTemplateModule ERROR - render_jinja_template function not found or not callable\n");
        PyGILState_Release(gil_state);
        PyThread_release_lock(g_init_lock);
        return -1;
    }
    Py_INCREF(g_render_jinja_template_func); // Keep a reference
    
    // Get the get_model_chat_template function
    g_get_model_chat_template_func = PyDict_GetItemString(module_dict, "get_model_chat_template");
    if (!g_get_model_chat_template_func || !PyCallable_Check(g_get_model_chat_template_func)) {
        printf("[C] Py_InitChatTemplateModule ERROR - get_model_chat_template function not found or not callable\n");
        PyGILState_Release(gil_state);
        PyThread_release_lock(g_init_lock);
        return -1;
    }
    Py_INCREF(g_get_model_chat_template_func); // Keep a reference
    
    // Release GIL
    PyGILState_Release(gil_state);
    
    g_initialized = 1;
    PyThread_release_lock(g_init_lock);
    return 0;
}



// Call the cached render_jinja_template function
char* Py_CallRenderJinjaTemplate(const char* json_request) {
    // Try direct call first (fast path)
    char* result = Py_CallRenderJinjaTemplateInternal(json_request);
    if (result != NULL) {
        return result;  // Success on first try
    }
    
    // If failed, just return NULL (no retry, no reload)
        return NULL;
}

// Internal function that does the actual work
char* Py_CallRenderJinjaTemplateInternal(const char* json_request) {    
    // Check if Python interpreter is still valid
    if (!Py_IsInitialized()) {
        printf("[C] Py_CallRenderJinjaTemplateInternal ERROR - Python interpreter not initialized\n");
        return NULL;
    }
    
    // Simple validation
    if (!json_request) {
        printf("[C] Py_CallRenderJinjaTemplateInternal ERROR - Input is NULL\n");
        return NULL;
    }
    
    // Acquire GIL for Python operations
    PyGILState_STATE gil_state = PyGILState_Ensure();    
    // Create Python string from JSON request
    PyObject* py_json = PyUnicode_FromString(json_request);
    if (!py_json) {
        printf("[C] Py_CallRenderJinjaTemplateInternal ERROR - Failed to create Python string\n");
        PyGILState_Release(gil_state);
        return NULL;
    }

    // Create arguments tuple
    PyObject* args = PyTuple_Pack(1, py_json);
    if (!args) {
        printf("[C] Py_CallRenderJinjaTemplateInternal ERROR - Failed to create args tuple\n");
        Py_DECREF(py_json);
        PyGILState_Release(gil_state);
        return NULL;
    }    

    // Call the cached function
    PyObject* py_result = PyObject_CallObject(g_render_jinja_template_func, args);
    
    // Clean up args
    Py_DECREF(args);
    Py_DECREF(py_json);
    
    char* cresult = NULL;
    if (py_result) {
        // Convert to C string
        const char* s = PyUnicode_AsUTF8(py_result);
        if (s) {
            cresult = strdup(s);
        } else {
            printf("[C] Py_CallRenderJinjaTemplateInternal ERROR - Failed to convert result to C string\n");
        }
        Py_DECREF(py_result);
    } else {
        printf("[C] Py_CallRenderJinjaTemplateInternal ERROR - Python function returned NULL\n");
        PyErr_Print();
        fflush(stderr);
    }
    
    // Release GIL
    PyGILState_Release(gil_state);
    
    return cresult;
}

// Call the cached get_model_chat_template function
char* Py_CallGetModelChatTemplate(const char* json_request) {    
    // Try direct call first (fast path)
    char* result = Py_CallGetModelChatTemplateInternal(json_request);
    if (result != NULL) {
        return result;  // Success on first try
    }
    
    // If failed, just return NULL (no retry, no reload)
    return NULL;
}

// Internal function that does the actual work
char* Py_CallGetModelChatTemplateInternal(const char* json_request) {    
    // Check if Python is initialized
    if (!g_python_initialized) {
        printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Python not initialized\n");
        fflush(stdout);
        return NULL;
    }
    
    // Validate cached function
    if (!g_get_model_chat_template_func) {
        printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Cached function is NULL\n");
        fflush(stdout);
        return NULL;
    }
    
    // Validate that the cached function is still a valid Python object
    fflush(stdout);
    if (!PyCallable_Check(g_get_model_chat_template_func)) {
        printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Cached function is not callable (corrupted?)\n");
        fflush(stdout);
        return NULL;
    }
    
    // Validate input
    if (!json_request) {
        printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Input is NULL\n");
        fflush(stdout);
        return NULL;
    }
    
    // Acquire GIL for Python operations
    PyGILState_STATE gil_state = PyGILState_Ensure();
    
    // Create Python string from JSON request
    PyObject* py_json = PyUnicode_FromString(json_request);
    if (!py_json) {
        printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Failed to create Python string\n");
        fflush(stdout);
        PyGILState_Release(gil_state);
        return NULL;
    }
    
    // Create arguments tuple
    PyObject* args = PyTuple_Pack(1, py_json);
    if (!args) {
        printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Failed to create args tuple\n");
        fflush(stdout);
        Py_DECREF(py_json);
        PyGILState_Release(gil_state);
        return NULL;
    }
    
    // Call the cached function
    PyObject* py_result = PyObject_CallObject(g_get_model_chat_template_func, args);
    
    // Clean up args
    Py_DECREF(args);
    Py_DECREF(py_json);
    
    char* cresult = NULL;
    if (py_result) {
        // Convert to C string
        const char* s = PyUnicode_AsUTF8(py_result);
        if (s) {
            cresult = strdup(s);
        } else {
            printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Failed to convert result to C string\n");
            fflush(stdout);
        }
        Py_DECREF(py_result);
    } else {
        printf("[C] Py_CallGetModelChatTemplateInternal ERROR - Python function returned NULL\n");
        fflush(stdout);
        PyErr_Print();
        fflush(stderr);
    }
    
    // Release GIL
    PyGILState_Release(gil_state);
    
    return cresult;
}

// Clear all caches for testing purposes
char* Py_ClearCaches() {
    if (!g_initialized) {
        printf("[C] Py_ClearCaches ERROR - Module not initialized\n");
        return NULL;
    }
    
    PyGILState_STATE gil_state = PyGILState_Ensure();
    
    // Call the clear_caches function
    PyObject* clear_caches_func = PyDict_GetItemString(PyModule_GetDict(g_chat_template_module), "clear_caches");
    if (!clear_caches_func || !PyCallable_Check(clear_caches_func)) {
        printf("[C] Py_ClearCaches ERROR - clear_caches function not found or not callable\n");
        PyGILState_Release(gil_state);
        return NULL;
    }
    
    PyObject* result = PyObject_CallObject(clear_caches_func, NULL);
    if (!result) {
        printf("[C] Py_ClearCaches ERROR - Failed to call clear_caches function\n");
        PyErr_Print();
        PyGILState_Release(gil_state);
        return NULL;
    }
    
    // Convert result to C string
    const char* result_str = PyUnicode_AsUTF8(result);
    if (!result_str) {
        printf("[C] Py_ClearCaches ERROR - Failed to convert result to string\n");
        Py_DECREF(result);
        PyGILState_Release(gil_state);
        return NULL;
    }
    
    char* c_result = strdup(result_str);
    Py_DECREF(result);
    PyGILState_Release(gil_state);
    
    return c_result;
}

// Clean up cached objects
void Py_CleanupChatTemplateModule() {
    if (g_initialized && Py_IsInitialized()) {
        PyGILState_STATE state = PyGILState_Ensure();
        Py_XDECREF(g_render_jinja_template_func);
        Py_XDECREF(g_get_model_chat_template_func);
        Py_XDECREF(g_chat_template_module);
        g_render_jinja_template_func = NULL;
        g_get_model_chat_template_func = NULL;
        g_chat_template_module = NULL;
        g_initialized = 0;
        PyGILState_Release(state);
    }
} 

// Re-initialize Python interpreter state
int Py_ReinitializeGo() {    
    // Reset global flags
    g_initialized = 0;
    g_python_initialized = 0;
    g_process_initialized = 0;
    
    // Clean up cached objects
    if (g_render_jinja_template_func) {
        Py_DECREF(g_render_jinja_template_func);
        g_render_jinja_template_func = NULL;
    }
    if (g_get_model_chat_template_func) {
        Py_DECREF(g_get_model_chat_template_func);
        g_get_model_chat_template_func = NULL;
    }
    if (g_chat_template_module) {
        Py_DECREF(g_chat_template_module);
        g_chat_template_module = NULL;
    }
        
    // Re-initialize
    int result = Py_InitializeGo();
    if (result != 0) {
        printf("[C] Py_ReinitializeGo ERROR - Failed to re-initialize Python\n");
        return result;
    }
    
    result = Py_InitChatTemplateModule();
    if (result != 0) {
        printf("[C] Py_ReinitializeGo ERROR - Failed to re-initialize chat template module\n");
        return result;
    }
    
    return 0;
}
