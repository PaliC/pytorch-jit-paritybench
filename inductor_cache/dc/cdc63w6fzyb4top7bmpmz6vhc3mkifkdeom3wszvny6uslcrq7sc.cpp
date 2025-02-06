
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(16L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(4L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = (-1L) + x1;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = static_cast<int64_t>(0);
                            auto tmp3 = tmp1 >= tmp2;
                            auto tmp4 = static_cast<int64_t>(4);
                            auto tmp5 = tmp1 < tmp4;
                            auto tmp6 = tmp3 & tmp5;
                            auto tmp7 = (-1L) + x2;
                            auto tmp8 = c10::convert<int64_t>(tmp7);
                            auto tmp9 = tmp8 >= tmp2;
                            auto tmp10 = tmp8 < tmp4;
                            auto tmp11 = tmp9 & tmp10;
                            auto tmp12 = tmp6 & tmp11;
                            auto tmp13 = [&]
                            {
                                auto tmp14 = in_ptr0[static_cast<int64_t>((-5L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp12 ? tmp13() : -std::numeric_limits<decltype(tmp13())>::infinity();
                            auto tmp16 = x2;
                            auto tmp17 = c10::convert<int64_t>(tmp16);
                            auto tmp18 = tmp17 >= tmp2;
                            auto tmp19 = tmp17 < tmp4;
                            auto tmp20 = tmp18 & tmp19;
                            auto tmp21 = tmp6 & tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr0[static_cast<int64_t>((-4L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp21 ? tmp22() : -std::numeric_limits<decltype(tmp22())>::infinity();
                            auto tmp25 = max_propagate_nan(tmp24, tmp15);
                            auto tmp26 = 1L + x2;
                            auto tmp27 = c10::convert<int64_t>(tmp26);
                            auto tmp28 = tmp27 >= tmp2;
                            auto tmp29 = tmp27 < tmp4;
                            auto tmp30 = tmp28 & tmp29;
                            auto tmp31 = tmp6 & tmp30;
                            auto tmp32 = [&]
                            {
                                auto tmp33 = in_ptr0[static_cast<int64_t>((-3L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp31 ? tmp32() : -std::numeric_limits<decltype(tmp32())>::infinity();
                            auto tmp35 = max_propagate_nan(tmp34, tmp25);
                            auto tmp36 = x1;
                            auto tmp37 = c10::convert<int64_t>(tmp36);
                            auto tmp38 = tmp37 >= tmp2;
                            auto tmp39 = tmp37 < tmp4;
                            auto tmp40 = tmp38 & tmp39;
                            auto tmp41 = tmp40 & tmp11;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr0[static_cast<int64_t>((-1L) + x2 + 4L*x1 + 16L*x0)];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp35);
                            auto tmp46 = tmp40 & tmp20;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr0[static_cast<int64_t>(x2 + 4L*x1 + 16L*x0)];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = tmp40 & tmp30;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr0[static_cast<int64_t>(1L + x2 + 4L*x1 + 16L*x0)];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp51 ? tmp52() : -std::numeric_limits<decltype(tmp52())>::infinity();
                            auto tmp55 = max_propagate_nan(tmp54, tmp50);
                            auto tmp56 = 1L + x1;
                            auto tmp57 = c10::convert<int64_t>(tmp56);
                            auto tmp58 = tmp57 >= tmp2;
                            auto tmp59 = tmp57 < tmp4;
                            auto tmp60 = tmp58 & tmp59;
                            auto tmp61 = tmp60 & tmp11;
                            auto tmp62 = [&]
                            {
                                auto tmp63 = in_ptr0[static_cast<int64_t>(3L + x2 + 4L*x1 + 16L*x0)];
                                return tmp63;
                            }
                            ;
                            auto tmp64 = tmp61 ? tmp62() : -std::numeric_limits<decltype(tmp62())>::infinity();
                            auto tmp65 = max_propagate_nan(tmp64, tmp55);
                            auto tmp66 = tmp60 & tmp20;
                            auto tmp67 = [&]
                            {
                                auto tmp68 = in_ptr0[static_cast<int64_t>(4L + x2 + 4L*x1 + 16L*x0)];
                                return tmp68;
                            }
                            ;
                            auto tmp69 = tmp66 ? tmp67() : -std::numeric_limits<decltype(tmp67())>::infinity();
                            auto tmp70 = max_propagate_nan(tmp69, tmp65);
                            auto tmp71 = tmp60 & tmp30;
                            auto tmp72 = [&]
                            {
                                auto tmp73 = in_ptr0[static_cast<int64_t>(5L + x2 + 4L*x1 + 16L*x0)];
                                return tmp73;
                            }
                            ;
                            auto tmp74 = tmp71 ? tmp72() : -std::numeric_limits<decltype(tmp72())>::infinity();
                            auto tmp75 = max_propagate_nan(tmp74, tmp70);
                            out_ptr0[static_cast<int64_t>(x2 + 4L*x1 + 16L*x0)] = tmp75;
                        }
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
        if(unlikely(PyTuple_GET_SIZE(args) != 2))
            throw std::runtime_error("requires 2 args");
        kernel(parse_arg<float*>(args, 0), parse_arg<float*>(args, 1));Py_RETURN_NONE;
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
