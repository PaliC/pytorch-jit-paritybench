
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp6 = in_ptr0[static_cast<int64_t>(x0)];
                        auto tmp9 = in_ptr1[static_cast<int64_t>(x0)];
                        auto tmp12 = in_ptr2[static_cast<int64_t>(x0)];
                        auto tmp14 = in_ptr3[static_cast<int64_t>(x0)];
                        auto tmp0 = x1;
                        auto tmp1 = c10::convert<int32_t>(tmp0);
                        auto tmp2 = static_cast<int32_t>(3);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp4 = static_cast<int32_t>(0);
                        auto tmp5 = tmp4 == tmp4;
                        auto tmp7 = static_cast<int32_t>(2);
                        auto tmp8 = tmp2 == tmp7;
                        auto tmp10 = static_cast<int32_t>(1);
                        auto tmp11 = tmp7 == tmp10;
                        auto tmp13 = tmp10 == tmp4;
                        auto tmp15 = static_cast<float>(0.0);
                        auto tmp16 = tmp5 ? tmp14 : tmp15;
                        auto tmp17 = tmp5 ? tmp16 : tmp15;
                        auto tmp18 = tmp13 ? tmp17 : tmp15;
                        auto tmp19 = tmp5 ? tmp12 : tmp18;
                        auto tmp20 = tmp5 ? tmp19 : tmp18;
                        auto tmp21 = tmp7 == tmp4;
                        auto tmp22 = tmp21 ? tmp17 : tmp15;
                        auto tmp23 = tmp11 ? tmp20 : tmp22;
                        auto tmp24 = tmp5 ? tmp9 : tmp23;
                        auto tmp25 = tmp5 ? tmp24 : tmp23;
                        auto tmp26 = tmp2 == tmp10;
                        auto tmp27 = tmp2 == tmp4;
                        auto tmp28 = tmp27 ? tmp17 : tmp15;
                        auto tmp29 = tmp26 ? tmp20 : tmp28;
                        auto tmp30 = tmp8 ? tmp25 : tmp29;
                        auto tmp31 = tmp5 ? tmp6 : tmp30;
                        auto tmp32 = tmp5 ? tmp31 : tmp30;
                        auto tmp33 = tmp1 == tmp7;
                        auto tmp34 = tmp1 == tmp10;
                        auto tmp35 = tmp1 == tmp4;
                        auto tmp36 = tmp35 ? tmp17 : tmp15;
                        auto tmp37 = tmp34 ? tmp20 : tmp36;
                        auto tmp38 = tmp33 ? tmp25 : tmp37;
                        auto tmp39 = tmp3 ? tmp32 : tmp38;
                        out_ptr0[static_cast<int64_t>(x1 + 4L*x0)] = tmp39;
                    }
                }
            }
        }
    }
}

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 5))
            throw std::runtime_error("requires 5 args");
        kernel(parse_arg<float*>(args, 0), parse_arg<float*>(args, 1), parse_arg<float*>(args, 2), parse_arg<float*>(args, 3), parse_arg<float*>(args, 4));Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(mod, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
