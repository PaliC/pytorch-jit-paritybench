
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const double* in_ptr0,
                       double* out_ptr0,
                       double* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x1;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = (static_cast<int64_t>((-1L) + x1) % static_cast<int64_t>(2L));
                        auto tmp5 = c10::convert<int64_t>(tmp4);
                        auto tmp6 = static_cast<int64_t>(0);
                        auto tmp7 = tmp5 == tmp6;
                        auto tmp8 = tmp3 & tmp7;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = tmp2 == tmp6;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = (static_cast<int64_t>(2L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + x1), static_cast<int64_t>(2L)))) % static_cast<int64_t>(2L));
                                auto tmp13 = c10::convert<int64_t>(tmp12);
                                auto tmp14 = tmp13 == tmp6;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = 2L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + x1), static_cast<int64_t>(2L)));
                                    auto tmp17 = c10::convert<int64_t>(tmp16);
                                    auto tmp18 = static_cast<int64_t>(2);
                                    auto tmp19 = tmp17 >= tmp18;
                                    auto tmp20 = [&]
                                    {
                                        auto tmp21 = in_ptr0[static_cast<int64_t>(x0)];
                                        auto tmp22 = static_cast<double>(16.0);
                                        auto tmp23 = decltype(tmp22)(tmp22 * tmp21);
                                        return tmp23;
                                    }
                                    ;
                                    auto tmp24 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                    auto tmp25 = static_cast<double>(0.0);
                                    auto tmp26 = tmp19 ? tmp24 : tmp25;
                                    auto tmp27 = tmp18 >= tmp18;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<int64_t>(x0)];
                                        auto tmp30 = static_cast<double>(16.0);
                                        auto tmp31 = decltype(tmp30)(tmp30 * tmp29);
                                        return tmp31;
                                    }
                                    ;
                                    auto tmp32 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    auto tmp33 = tmp27 ? tmp32 : tmp25;
                                    auto tmp34 = static_cast<double>(0.5);
                                    auto tmp35 = decltype(tmp33)(tmp33 * tmp34);
                                    auto tmp36 = decltype(tmp26)(tmp26 - tmp35);
                                    return tmp36;
                                }
                                ;
                                auto tmp37 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                auto tmp38 = 2L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + x1), static_cast<int64_t>(2L)));
                                auto tmp39 = c10::convert<int64_t>(tmp38);
                                auto tmp40 = static_cast<int64_t>(2);
                                auto tmp41 = tmp39 >= tmp40;
                                auto tmp42 = [&]
                                {
                                    auto tmp43 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp44 = static_cast<double>(16.0);
                                    auto tmp45 = decltype(tmp44)(tmp44 * tmp43);
                                    return tmp45;
                                }
                                ;
                                auto tmp46 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                                auto tmp47 = static_cast<double>(0.0);
                                auto tmp48 = tmp41 ? tmp46 : tmp47;
                                auto tmp49 = tmp14 ? tmp37 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp51 = [&]
                            {
                                auto tmp52 = 2L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + x1), static_cast<int64_t>(2L)));
                                auto tmp53 = c10::convert<int64_t>(tmp52);
                                auto tmp54 = static_cast<int64_t>(2);
                                auto tmp55 = tmp53 >= tmp54;
                                auto tmp56 = [&]
                                {
                                    auto tmp57 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp58 = static_cast<double>(16.0);
                                    auto tmp59 = decltype(tmp58)(tmp58 * tmp57);
                                    return tmp59;
                                }
                                ;
                                auto tmp60 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                                auto tmp61 = static_cast<double>(0.0);
                                auto tmp62 = tmp55 ? tmp60 : tmp61;
                                auto tmp63 = tmp54 >= tmp54;
                                auto tmp64 = [&]
                                {
                                    auto tmp65 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp66 = static_cast<double>(16.0);
                                    auto tmp67 = decltype(tmp66)(tmp66 * tmp65);
                                    return tmp67;
                                }
                                ;
                                auto tmp68 = tmp63 ? tmp64() : static_cast<decltype(tmp64())>(0.0);
                                auto tmp69 = tmp63 ? tmp68 : tmp61;
                                auto tmp70 = static_cast<double>(0.5);
                                auto tmp71 = decltype(tmp69)(tmp69 * tmp70);
                                auto tmp72 = decltype(tmp62)(tmp62 - tmp71);
                                return tmp72;
                            }
                            ;
                            auto tmp73 = tmp10 ? tmp51() : static_cast<decltype(tmp51())>(0.0);
                            auto tmp74 = 1L + 2L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + x1), static_cast<int64_t>(2L)));
                            auto tmp75 = c10::convert<int64_t>(tmp74);
                            auto tmp76 = static_cast<int64_t>(2);
                            auto tmp77 = tmp75 >= tmp76;
                            auto tmp78 = [&]
                            {
                                auto tmp79 = in_ptr0[static_cast<int64_t>(x0)];
                                auto tmp80 = static_cast<double>(16.0);
                                auto tmp81 = decltype(tmp80)(tmp80 * tmp79);
                                return tmp81;
                            }
                            ;
                            auto tmp82 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                            auto tmp83 = static_cast<double>(0.0);
                            auto tmp84 = tmp77 ? tmp82 : tmp83;
                            auto tmp85 = tmp10 ? tmp73 : tmp84;
                            auto tmp86 = tmp10 ? tmp50 : tmp85;
                            auto tmp87 = [&]
                            {
                                auto tmp88 = tmp6 == tmp6;
                                auto tmp89 = [&]
                                {
                                    auto tmp90 = tmp76 >= tmp76;
                                    auto tmp91 = [&]
                                    {
                                        auto tmp92 = in_ptr0[static_cast<int64_t>(x0)];
                                        auto tmp93 = static_cast<double>(16.0);
                                        auto tmp94 = decltype(tmp93)(tmp93 * tmp92);
                                        return tmp94;
                                    }
                                    ;
                                    auto tmp95 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(0.0);
                                    auto tmp96 = tmp90 ? tmp95 : tmp83;
                                    auto tmp97 = [&]
                                    {
                                        auto tmp98 = in_ptr0[static_cast<int64_t>(x0)];
                                        auto tmp99 = static_cast<double>(16.0);
                                        auto tmp100 = decltype(tmp99)(tmp99 * tmp98);
                                        return tmp100;
                                    }
                                    ;
                                    auto tmp101 = tmp90 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                                    auto tmp102 = tmp90 ? tmp101 : tmp83;
                                    auto tmp103 = static_cast<double>(0.5);
                                    auto tmp104 = decltype(tmp102)(tmp102 * tmp103);
                                    auto tmp105 = decltype(tmp96)(tmp96 - tmp104);
                                    return tmp105;
                                }
                                ;
                                auto tmp106 = tmp88 ? tmp89() : static_cast<decltype(tmp89())>(0.0);
                                auto tmp107 = tmp76 >= tmp76;
                                auto tmp108 = [&]
                                {
                                    auto tmp109 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp110 = static_cast<double>(16.0);
                                    auto tmp111 = decltype(tmp110)(tmp110 * tmp109);
                                    return tmp111;
                                }
                                ;
                                auto tmp112 = tmp107 ? tmp108() : static_cast<decltype(tmp108())>(0.0);
                                auto tmp113 = tmp107 ? tmp112 : tmp83;
                                auto tmp114 = tmp88 ? tmp106 : tmp113;
                                return tmp114;
                            }
                            ;
                            auto tmp115 = tmp10 ? tmp87() : static_cast<decltype(tmp87())>(0.0);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = tmp76 >= tmp76;
                                auto tmp118 = [&]
                                {
                                    auto tmp119 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp120 = static_cast<double>(16.0);
                                    auto tmp121 = decltype(tmp120)(tmp120 * tmp119);
                                    return tmp121;
                                }
                                ;
                                auto tmp122 = tmp117 ? tmp118() : static_cast<decltype(tmp118())>(0.0);
                                auto tmp123 = tmp117 ? tmp122 : tmp83;
                                auto tmp124 = [&]
                                {
                                    auto tmp125 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp126 = static_cast<double>(16.0);
                                    auto tmp127 = decltype(tmp126)(tmp126 * tmp125);
                                    return tmp127;
                                }
                                ;
                                auto tmp128 = tmp117 ? tmp124() : static_cast<decltype(tmp124())>(0.0);
                                auto tmp129 = tmp117 ? tmp128 : tmp83;
                                auto tmp130 = static_cast<double>(0.5);
                                auto tmp131 = decltype(tmp129)(tmp129 * tmp130);
                                auto tmp132 = decltype(tmp123)(tmp123 - tmp131);
                                return tmp132;
                            }
                            ;
                            auto tmp133 = tmp10 ? tmp116() : static_cast<decltype(tmp116())>(0.0);
                            auto tmp134 = static_cast<int64_t>(3);
                            auto tmp135 = tmp134 >= tmp76;
                            auto tmp136 = [&]
                            {
                                auto tmp137 = in_ptr0[static_cast<int64_t>(x0)];
                                auto tmp138 = static_cast<double>(16.0);
                                auto tmp139 = decltype(tmp138)(tmp138 * tmp137);
                                return tmp139;
                            }
                            ;
                            auto tmp140 = tmp135 ? tmp136() : static_cast<decltype(tmp136())>(0.0);
                            auto tmp141 = tmp135 ? tmp140 : tmp83;
                            auto tmp142 = tmp10 ? tmp133 : tmp141;
                            auto tmp143 = tmp10 ? tmp115 : tmp142;
                            auto tmp144 = static_cast<double>(0.5);
                            auto tmp145 = decltype(tmp143)(tmp143 * tmp144);
                            auto tmp146 = decltype(tmp86)(tmp86 - tmp145);
                            return tmp146;
                        }
                        ;
                        auto tmp147 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp148 = (static_cast<int64_t>(x1) % static_cast<int64_t>(2L));
                        auto tmp149 = c10::convert<int64_t>(tmp148);
                        auto tmp150 = tmp149 == tmp6;
                        auto tmp151 = [&]
                        {
                            auto tmp152 = (static_cast<int64_t>(2L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L)))) % static_cast<int64_t>(2L));
                            auto tmp153 = c10::convert<int64_t>(tmp152);
                            auto tmp154 = tmp153 == tmp6;
                            auto tmp155 = [&]
                            {
                                auto tmp156 = 2L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L)));
                                auto tmp157 = c10::convert<int64_t>(tmp156);
                                auto tmp158 = static_cast<int64_t>(2);
                                auto tmp159 = tmp157 >= tmp158;
                                auto tmp160 = [&]
                                {
                                    auto tmp161 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp162 = static_cast<double>(16.0);
                                    auto tmp163 = decltype(tmp162)(tmp162 * tmp161);
                                    return tmp163;
                                }
                                ;
                                auto tmp164 = tmp159 ? tmp160() : static_cast<decltype(tmp160())>(0.0);
                                auto tmp165 = static_cast<double>(0.0);
                                auto tmp166 = tmp159 ? tmp164 : tmp165;
                                auto tmp167 = tmp158 >= tmp158;
                                auto tmp168 = [&]
                                {
                                    auto tmp169 = in_ptr0[static_cast<int64_t>(x0)];
                                    auto tmp170 = static_cast<double>(16.0);
                                    auto tmp171 = decltype(tmp170)(tmp170 * tmp169);
                                    return tmp171;
                                }
                                ;
                                auto tmp172 = tmp167 ? tmp168() : static_cast<decltype(tmp168())>(0.0);
                                auto tmp173 = tmp167 ? tmp172 : tmp165;
                                auto tmp174 = static_cast<double>(0.5);
                                auto tmp175 = decltype(tmp173)(tmp173 * tmp174);
                                auto tmp176 = decltype(tmp166)(tmp166 - tmp175);
                                return tmp176;
                            }
                            ;
                            auto tmp177 = tmp154 ? tmp155() : static_cast<decltype(tmp155())>(0.0);
                            auto tmp178 = 2L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L)));
                            auto tmp179 = c10::convert<int64_t>(tmp178);
                            auto tmp180 = static_cast<int64_t>(2);
                            auto tmp181 = tmp179 >= tmp180;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = in_ptr0[static_cast<int64_t>(x0)];
                                auto tmp184 = static_cast<double>(16.0);
                                auto tmp185 = decltype(tmp184)(tmp184 * tmp183);
                                return tmp185;
                            }
                            ;
                            auto tmp186 = tmp181 ? tmp182() : static_cast<decltype(tmp182())>(0.0);
                            auto tmp187 = static_cast<double>(0.0);
                            auto tmp188 = tmp181 ? tmp186 : tmp187;
                            auto tmp189 = tmp154 ? tmp177 : tmp188;
                            return tmp189;
                        }
                        ;
                        auto tmp190 = tmp150 ? tmp151() : static_cast<decltype(tmp151())>(0.0);
                        auto tmp191 = [&]
                        {
                            auto tmp192 = 2L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(2L)));
                            auto tmp193 = c10::convert<int64_t>(tmp192);
                            auto tmp194 = static_cast<int64_t>(2);
                            auto tmp195 = tmp193 >= tmp194;
                            auto tmp196 = [&]
                            {
                                auto tmp197 = in_ptr0[static_cast<int64_t>(x0)];
                                auto tmp198 = static_cast<double>(16.0);
                                auto tmp199 = decltype(tmp198)(tmp198 * tmp197);
                                return tmp199;
                            }
                            ;
                            auto tmp200 = tmp195 ? tmp196() : static_cast<decltype(tmp196())>(0.0);
                            auto tmp201 = static_cast<double>(0.0);
                            auto tmp202 = tmp195 ? tmp200 : tmp201;
                            auto tmp203 = tmp194 >= tmp194;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = in_ptr0[static_cast<int64_t>(x0)];
                                auto tmp206 = static_cast<double>(16.0);
                                auto tmp207 = decltype(tmp206)(tmp206 * tmp205);
                                return tmp207;
                            }
                            ;
                            auto tmp208 = tmp203 ? tmp204() : static_cast<decltype(tmp204())>(0.0);
                            auto tmp209 = tmp203 ? tmp208 : tmp201;
                            auto tmp210 = static_cast<double>(0.5);
                            auto tmp211 = decltype(tmp209)(tmp209 * tmp210);
                            auto tmp212 = decltype(tmp202)(tmp202 - tmp211);
                            return tmp212;
                        }
                        ;
                        auto tmp213 = tmp150 ? tmp191() : static_cast<decltype(tmp191())>(0.0);
                        auto tmp214 = static_cast<int64_t>(2);
                        auto tmp215 = tmp1 >= tmp214;
                        auto tmp216 = [&]
                        {
                            auto tmp217 = in_ptr0[static_cast<int64_t>(x0)];
                            auto tmp218 = static_cast<double>(16.0);
                            auto tmp219 = decltype(tmp218)(tmp218 * tmp217);
                            return tmp219;
                        }
                        ;
                        auto tmp220 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                        auto tmp221 = static_cast<double>(0.0);
                        auto tmp222 = tmp215 ? tmp220 : tmp221;
                        auto tmp223 = tmp150 ? tmp213 : tmp222;
                        auto tmp224 = tmp150 ? tmp190 : tmp223;
                        auto tmp225 = tmp8 ? tmp147 : tmp224;
                        out_ptr0[static_cast<int64_t>(x1 + 4L*x0)] = tmp225;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp12 = out_ptr0[static_cast<int64_t>(x1 + 4L*x0)];
                        auto tmp0 = x1;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = (static_cast<int64_t>((-1L) + x1) % static_cast<int64_t>(2L));
                        auto tmp5 = c10::convert<int64_t>(tmp4);
                        auto tmp6 = static_cast<int64_t>(0);
                        auto tmp7 = tmp5 == tmp6;
                        auto tmp8 = tmp3 & tmp7;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = out_ptr0[static_cast<int64_t>(1L + 2L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + x1), static_cast<int64_t>(2L))) + 4L*x0)];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp13 = tmp8 ? tmp11 : tmp12;
                        out_ptr1[static_cast<int64_t>(x1 + 4L*x0)] = tmp13;
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
        if(unlikely(PyTuple_GET_SIZE(args) != 3))
            throw std::runtime_error("requires 3 args");
        kernel(parse_arg<double*>(args, 0), parse_arg<double*>(args, 1), parse_arg<double*>(args, 2));Py_RETURN_NONE;
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
