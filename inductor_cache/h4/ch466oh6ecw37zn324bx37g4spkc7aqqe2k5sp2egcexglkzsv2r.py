# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: inductor_cache/7a/c7aqbkfl7wcdeejqihzlytki6mvajrsnfle6btggat5vjuc7mf75.py
# Topologically Sorted Source Nodes: [add, tile_disp, mul_1, add_1, mul_2, add_2, mul_3, add_3, mul_4, add_4, mul_5, add_5, mul_6, add_6, mul_7, add_7, mul_8, add_8, mul_9, add_9, mul_10, add_10, mul_11, add_11, mul_12, add_12, mul_13, add_13, mul_14, add_14, mul_15, add_15, mul_16, add_16, mul_17, add_17, mul_18, add_18, mul_19, add_19, mul_20, add_20, mul_21, add_21, mul_22, add_22, mul_23, add_23, mul_24, add_24, mul_25, add_25, mul_26, add_26, mul_27, add_27, mul_28, add_28, mul_29, add_29, mul_30, add_30, mul_31, add_31, mul_32, add_32, add_33, tile_disp_1, mul_36, add_34, mul_37, add_35, mul_38, add_36, mul_39, add_37, mul_40, add_38, mul_41, add_39, mul_42, add_40, mul_43, add_41, mul_44, add_42, mul_45, add_43, mul_46, add_44, mul_47, add_45, mul_48, add_46, mul_49, add_47, mul_50, add_48, mul_51, add_49, mul_52, add_50, mul_53, add_51, mul_54, add_52, mul_55, add_53, mul_56, add_54, mul_57, add_55, mul_58, add_56, mul_59, add_57, mul_60, add_58, mul_61, add_59, mul_62, add_60, mul_63, add_61, mul_64, add_62, mul_65, add_63, mul_66, add_64, mul_67, add_65, add_66, tile_disp_2, mul_71, add_67, mul_72, add_68, mul_73, add_69, mul_74, add_70, mul_75, add_71, mul_76, add_72, mul_77, add_73, mul_78, add_74, mul_79, add_75, mul_80, add_76, mul_81, add_77, mul_82, add_78, mul_83, add_79, mul_84, add_80, mul_85, add_81, mul_86, add_82, mul_87, add_83, mul_88, add_84, mul_89, add_85, mul_90, add_86, mul_91, add_87, mul_92, add_88, mul_93, add_89, mul_94, add_90, mul_95, add_91, mul_96, add_92, mul_97, add_93, mul_98, add_94, mul_99, add_95, mul_100, add_96, mul_101, add_97, mul_102, add_98], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_10 => add_10
#   add_11 => add_11
#   add_12 => add_12
#   add_13 => add_13
#   add_14 => add_14
#   add_15 => add_15
#   add_16 => add_16
#   add_17 => add_17
#   add_18 => add_18
#   add_19 => add_19
#   add_2 => add_2
#   add_20 => add_20
#   add_21 => add_21
#   add_22 => add_22
#   add_23 => add_23
#   add_24 => add_24
#   add_25 => add_25
#   add_26 => add_26
#   add_27 => add_27
#   add_28 => add_28
#   add_29 => add_29
#   add_3 => add_3
#   add_30 => add_30
#   add_31 => add_31
#   add_32 => add_32
#   add_33 => add_40
#   add_34 => add_41
#   add_35 => add_42
#   add_36 => add_43
#   add_37 => add_44
#   add_38 => add_45
#   add_39 => add_46
#   add_4 => add_4
#   add_40 => add_47
#   add_41 => add_48
#   add_42 => add_49
#   add_43 => add_50
#   add_44 => add_51
#   add_45 => add_52
#   add_46 => add_53
#   add_47 => add_54
#   add_48 => add_55
#   add_49 => add_56
#   add_5 => add_5
#   add_50 => add_57
#   add_51 => add_58
#   add_52 => add_59
#   add_53 => add_60
#   add_54 => add_61
#   add_55 => add_62
#   add_56 => add_63
#   add_57 => add_64
#   add_58 => add_65
#   add_59 => add_66
#   add_6 => add_6
#   add_60 => add_67
#   add_61 => add_68
#   add_62 => add_69
#   add_63 => add_70
#   add_64 => add_71
#   add_65 => add_72
#   add_66 => add_80
#   add_67 => add_81
#   add_68 => add_82
#   add_69 => add_83
#   add_7 => add_7
#   add_70 => add_84
#   add_71 => add_85
#   add_72 => add_86
#   add_73 => add_87
#   add_74 => add_88
#   add_75 => add_89
#   add_76 => add_90
#   add_77 => add_91
#   add_78 => add_92
#   add_79 => add_93
#   add_8 => add_8
#   add_80 => add_94
#   add_81 => add_95
#   add_82 => add_96
#   add_83 => add_97
#   add_84 => add_98
#   add_85 => add_99
#   add_86 => add_100
#   add_87 => add_101
#   add_88 => add_102
#   add_89 => add_103
#   add_9 => add_9
#   add_90 => add_104
#   add_91 => add_105
#   add_92 => add_106
#   add_93 => add_107
#   add_94 => add_108
#   add_95 => add_109
#   add_96 => add_110
#   add_97 => add_111
#   add_98 => add_112
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_100 => mul_120
#   mul_101 => mul_121
#   mul_102 => mul_122
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_3 => mul_3
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_36 => mul_46
#   mul_37 => mul_47
#   mul_38 => mul_48
#   mul_39 => mul_49
#   mul_4 => mul_4
#   mul_40 => mul_50
#   mul_41 => mul_51
#   mul_42 => mul_52
#   mul_43 => mul_53
#   mul_44 => mul_54
#   mul_45 => mul_55
#   mul_46 => mul_56
#   mul_47 => mul_57
#   mul_48 => mul_58
#   mul_49 => mul_59
#   mul_5 => mul_5
#   mul_50 => mul_60
#   mul_51 => mul_61
#   mul_52 => mul_62
#   mul_53 => mul_63
#   mul_54 => mul_64
#   mul_55 => mul_65
#   mul_56 => mul_66
#   mul_57 => mul_67
#   mul_58 => mul_68
#   mul_59 => mul_69
#   mul_6 => mul_6
#   mul_60 => mul_70
#   mul_61 => mul_71
#   mul_62 => mul_72
#   mul_63 => mul_73
#   mul_64 => mul_74
#   mul_65 => mul_75
#   mul_66 => mul_76
#   mul_67 => mul_77
#   mul_7 => mul_7
#   mul_71 => mul_91
#   mul_72 => mul_92
#   mul_73 => mul_93
#   mul_74 => mul_94
#   mul_75 => mul_95
#   mul_76 => mul_96
#   mul_77 => mul_97
#   mul_78 => mul_98
#   mul_79 => mul_99
#   mul_8 => mul_8
#   mul_80 => mul_100
#   mul_81 => mul_101
#   mul_82 => mul_102
#   mul_83 => mul_103
#   mul_84 => mul_104
#   mul_85 => mul_105
#   mul_86 => mul_106
#   mul_87 => mul_107
#   mul_88 => mul_108
#   mul_89 => mul_109
#   mul_9 => mul_9
#   mul_90 => mul_110
#   mul_91 => mul_111
#   mul_92 => mul_112
#   mul_93 => mul_113
#   mul_94 => mul_114
#   mul_95 => mul_115
#   mul_96 => mul_116
#   mul_97 => mul_117
#   mul_98 => mul_118
#   mul_99 => mul_119
#   tile_disp => mul
#   tile_disp_1 => mul_45
#   tile_disp_2 => mul_90
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, -1), kwargs = {})
#   %mul : [num_users=16] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mul_8), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_12), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %mul_14), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_15), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %mul_16), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_17), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %mul_18), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_19), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %mul_20), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_21), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %mul_22), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_23), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %mul_24), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_25), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %mul_26), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_27), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %mul_28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_29), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %mul_30), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_31), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %mul_32), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, 0), kwargs = {})
#   %mul_45 : [num_users=16] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 1.0), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_46), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %mul_47), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_48), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %mul_49), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_50), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %mul_51), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_52), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_47, %mul_53), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_54), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %mul_55), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_56), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_51, %mul_57), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_58), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_53, %mul_59), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_60), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %mul_61), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_62), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %mul_63), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_64), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_59, %mul_65), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_66), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_61, %mul_67), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_68), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, %mul_69), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_70), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %mul_71), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_72), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_67, %mul_73), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_74), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %mul_75), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_76), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %mul_77), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %mul_90 : [num_users=16] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_80, 1.0), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_91), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_81, %mul_92), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_93), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_83, %mul_94), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_95), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_85, %mul_96), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -1.5), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_97), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_87, %mul_98), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_99), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_89, %mul_100), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_101), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %mul_102), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_103), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_93, %mul_104), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, -0.5), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_105), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_95, %mul_106), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_107), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_97, %mul_108), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_109), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_99, %mul_110), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_101 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_111), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_101, %mul_112), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.5), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_113), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_103, %mul_114), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_115), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -1.5), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_105, %mul_116), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_117), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, -0.5), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_107, %mul_118), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_119), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 0.5), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, %mul_120), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.5), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %mul_121), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, 1.5), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_111, %mul_122), kwargs = {})
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'out_ptr27': '*fp32', 'out_ptr28': '*fp32', 'out_ptr29': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32', 'out_ptr36': '*fp32', 'out_ptr37': '*fp32', 'out_ptr38': '*fp32', 'out_ptr39': '*fp32', 'out_ptr40': '*fp32', 'out_ptr41': '*fp32', 'out_ptr42': '*fp32', 'out_ptr43': '*fp32', 'out_ptr44': '*fp32', 'out_ptr45': '*fp32', 'out_ptr46': '*fp32', 'out_ptr47': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 17, 33, 49), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, out_ptr42, out_ptr43, out_ptr44, out_ptr45, out_ptr46, out_ptr47, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp1 = -1.0
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp6 = -1.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp10 = tmp9 * tmp6
    tmp11 = tmp8 + tmp10
    tmp12 = -0.5
    tmp13 = tmp9 * tmp12
    tmp14 = tmp8 + tmp13
    tmp15 = 0.5
    tmp16 = tmp9 * tmp15
    tmp17 = tmp8 + tmp16
    tmp18 = 1.5
    tmp19 = tmp9 * tmp18
    tmp20 = tmp8 + tmp19
    tmp21 = tmp5 * tmp12
    tmp22 = tmp4 + tmp21
    tmp23 = tmp22 + tmp10
    tmp24 = tmp22 + tmp13
    tmp25 = tmp22 + tmp16
    tmp26 = tmp22 + tmp19
    tmp27 = tmp5 * tmp15
    tmp28 = tmp4 + tmp27
    tmp29 = tmp28 + tmp10
    tmp30 = tmp28 + tmp13
    tmp31 = tmp28 + tmp16
    tmp32 = tmp28 + tmp19
    tmp33 = tmp5 * tmp18
    tmp34 = tmp4 + tmp33
    tmp35 = tmp34 + tmp10
    tmp36 = tmp34 + tmp13
    tmp37 = tmp34 + tmp16
    tmp38 = tmp34 + tmp19
    tmp39 = 0.0
    tmp40 = tmp0 + tmp39
    tmp41 = tmp40 * tmp3
    tmp42 = tmp41 + tmp7
    tmp43 = tmp42 + tmp10
    tmp44 = tmp42 + tmp13
    tmp45 = tmp42 + tmp16
    tmp46 = tmp42 + tmp19
    tmp47 = tmp41 + tmp21
    tmp48 = tmp47 + tmp10
    tmp49 = tmp47 + tmp13
    tmp50 = tmp47 + tmp16
    tmp51 = tmp47 + tmp19
    tmp52 = tmp41 + tmp27
    tmp53 = tmp52 + tmp10
    tmp54 = tmp52 + tmp13
    tmp55 = tmp52 + tmp16
    tmp56 = tmp52 + tmp19
    tmp57 = tmp41 + tmp33
    tmp58 = tmp57 + tmp10
    tmp59 = tmp57 + tmp13
    tmp60 = tmp57 + tmp16
    tmp61 = tmp57 + tmp19
    tmp62 = tmp0 + tmp3
    tmp63 = tmp62 * tmp3
    tmp64 = tmp63 + tmp7
    tmp65 = tmp64 + tmp10
    tmp66 = tmp64 + tmp13
    tmp67 = tmp64 + tmp16
    tmp68 = tmp64 + tmp19
    tmp69 = tmp63 + tmp21
    tmp70 = tmp69 + tmp10
    tmp71 = tmp69 + tmp13
    tmp72 = tmp69 + tmp16
    tmp73 = tmp69 + tmp19
    tmp74 = tmp63 + tmp27
    tmp75 = tmp74 + tmp10
    tmp76 = tmp74 + tmp13
    tmp77 = tmp74 + tmp16
    tmp78 = tmp74 + tmp19
    tmp79 = tmp63 + tmp33
    tmp80 = tmp79 + tmp10
    tmp81 = tmp79 + tmp13
    tmp82 = tmp79 + tmp16
    tmp83 = tmp79 + tmp19
    tl.store(out_ptr0 + (16*x2), tmp11, xmask)
    tl.store(out_ptr1 + (16*x2), tmp14, xmask)
    tl.store(out_ptr2 + (16*x2), tmp17, xmask)
    tl.store(out_ptr3 + (16*x2), tmp20, xmask)
    tl.store(out_ptr4 + (16*x2), tmp23, xmask)
    tl.store(out_ptr5 + (16*x2), tmp24, xmask)
    tl.store(out_ptr6 + (16*x2), tmp25, xmask)
    tl.store(out_ptr7 + (16*x2), tmp26, xmask)
    tl.store(out_ptr8 + (16*x2), tmp29, xmask)
    tl.store(out_ptr9 + (16*x2), tmp30, xmask)
    tl.store(out_ptr10 + (16*x2), tmp31, xmask)
    tl.store(out_ptr11 + (16*x2), tmp32, xmask)
    tl.store(out_ptr12 + (16*x2), tmp35, xmask)
    tl.store(out_ptr13 + (16*x2), tmp36, xmask)
    tl.store(out_ptr14 + (16*x2), tmp37, xmask)
    tl.store(out_ptr15 + (16*x2), tmp38, xmask)
    tl.store(out_ptr16 + (16*x2), tmp43, xmask)
    tl.store(out_ptr17 + (16*x2), tmp44, xmask)
    tl.store(out_ptr18 + (16*x2), tmp45, xmask)
    tl.store(out_ptr19 + (16*x2), tmp46, xmask)
    tl.store(out_ptr20 + (16*x2), tmp48, xmask)
    tl.store(out_ptr21 + (16*x2), tmp49, xmask)
    tl.store(out_ptr22 + (16*x2), tmp50, xmask)
    tl.store(out_ptr23 + (16*x2), tmp51, xmask)
    tl.store(out_ptr24 + (16*x2), tmp53, xmask)
    tl.store(out_ptr25 + (16*x2), tmp54, xmask)
    tl.store(out_ptr26 + (16*x2), tmp55, xmask)
    tl.store(out_ptr27 + (16*x2), tmp56, xmask)
    tl.store(out_ptr28 + (16*x2), tmp58, xmask)
    tl.store(out_ptr29 + (16*x2), tmp59, xmask)
    tl.store(out_ptr30 + (16*x2), tmp60, xmask)
    tl.store(out_ptr31 + (16*x2), tmp61, xmask)
    tl.store(out_ptr32 + (16*x2), tmp65, xmask)
    tl.store(out_ptr33 + (16*x2), tmp66, xmask)
    tl.store(out_ptr34 + (16*x2), tmp67, xmask)
    tl.store(out_ptr35 + (16*x2), tmp68, xmask)
    tl.store(out_ptr36 + (16*x2), tmp70, xmask)
    tl.store(out_ptr37 + (16*x2), tmp71, xmask)
    tl.store(out_ptr38 + (16*x2), tmp72, xmask)
    tl.store(out_ptr39 + (16*x2), tmp73, xmask)
    tl.store(out_ptr40 + (16*x2), tmp75, xmask)
    tl.store(out_ptr41 + (16*x2), tmp76, xmask)
    tl.store(out_ptr42 + (16*x2), tmp77, xmask)
    tl.store(out_ptr43 + (16*x2), tmp78, xmask)
    tl.store(out_ptr44 + (16*x2), tmp80, xmask)
    tl.store(out_ptr45 + (16*x2), tmp81, xmask)
    tl.store(out_ptr46 + (16*x2), tmp82, xmask)
    tl.store(out_ptr47 + (16*x2), tmp83, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gt/cgtva3k4cdam2dhtmeghkwdkbyfer64imwmkzeai3xeh5na6jpkr.py
# Topologically Sorted Source Nodes: [cat_1, vgrid, sub, setitem, clone, mul_33, truediv, sub_1, setitem_1], Original ATen: [aten.cat, aten._to_copy, aten.sub, aten.copy, aten.clone, aten.mul, aten.div]
# Source node to ATen node mapping:
#   cat_1 => cat_1
#   clone => clone_2
#   mul_33 => mul_33
#   setitem => copy
#   setitem_1 => copy_1
#   sub => sub
#   sub_1 => sub_1
#   truediv => div
#   vgrid => convert_element_type
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_4, %repeat_5], 1), kwargs = {})
#   %convert_element_type : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_19, %view_2), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_23, %sub), kwargs = {})
#   %slice_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%convert_element_type, %copy, 1, 0, 1), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_4,), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_2, 2.0), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_33, 3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, 1.0), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_6, %sub_1), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default, %copy_1, 1, 0), kwargs = {})
triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1 = async_compile.triton('triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 2)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 32
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp6 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = x0
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tmp6 >= tmp8
    tmp15 = tl.full([1], 2, tl.int64)
    tmp16 = tmp6 < tmp15
    tmp17 = tmp14 & tmp5
    tmp18 = x1
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tl.where(tmp9, tmp13, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr0 + (x0 + 4*((((x3 % 16)) % 4)) + 16*x1 + 64*(((x3 % 16)) // 4) + 256*(x3 // 16)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 - tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp5, tmp24, tmp25)
    tmp27 = tmp3 >= tmp3
    tmp28 = x0
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tmp3 >= tmp4
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp3 < tmp32
    tmp34 = x1
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tl.where(tmp5, tmp30, tmp36)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.where(tmp5, tmp26, tmp38)
    tmp40 = 2.0
    tmp41 = tmp39 * tmp40
    tmp42 = 0.3333333333333333
    tmp43 = tmp41 * tmp42
    tmp44 = 1.0
    tmp45 = tmp43 - tmp44
    tmp46 = tmp0 < tmp4
    tmp47 = x2
    tmp48 = tl.full([1], 0, tl.int64)
    tmp49 = tmp47 >= tmp48
    tmp50 = tl.full([1], 1, tl.int64)
    tmp51 = tmp47 < tmp50
    tmp52 = tmp51 & tmp46
    tmp53 = x0
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp47 >= tmp50
    tmp57 = tl.full([1], 2, tl.int64)
    tmp58 = tmp47 < tmp57
    tmp59 = tmp56 & tmp46
    tmp60 = x1
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = tl.where(tmp51, tmp55, tmp62)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tl.load(in_ptr0 + (x0 + 4*((((x3 % 16)) % 4)) + 16*x1 + 64*(((x3 % 16)) // 4) + 256*(x3 // 16)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 - tmp65
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp46, tmp66, tmp67)
    tmp69 = tmp0 >= tmp3
    tmp70 = x0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp46, tmp70, tmp71)
    tmp73 = tmp0 >= tmp4
    tmp74 = tmp0 < tmp32
    tmp75 = x1
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp73, tmp75, tmp76)
    tmp78 = tl.where(tmp46, tmp72, tmp77)
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tl.where(tmp46, tmp68, tmp79)
    tmp81 = tl.where(tmp2, tmp45, tmp80)
    tl.store(out_ptr0 + (x4), tmp81, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jv/cjvm2muod2elpjas7ev5q57kdw2vo2g7hjq2t73v3oe3ia5cabha.py
# Topologically Sorted Source Nodes: [output, output_1], Original ATen: [aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   output => add_33, add_34, add_35, add_36, convert_element_type_1, convert_element_type_2, convert_element_type_8, floor, floor_1, full_default, full_default_1, full_default_10, full_default_11, full_default_2, full_default_5, full_default_8, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index_1, index_2, index_3, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt, lt_1, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, mul_35, mul_36, mul_37, mul_38, mul_39, mul_40, mul_42, mul_43, mul_44, sub_10, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9, where, where_1, where_10, where_11, where_2, where_5, where_8
#   output_1 => add_73, add_74, add_75, add_76, convert_element_type_10, convert_element_type_11, convert_element_type_17, floor_2, floor_3, full_default_12, full_default_13, full_default_14, full_default_17, full_default_20, full_default_22, full_default_23, ge_10, ge_11, ge_12, ge_13, ge_14, ge_15, ge_8, ge_9, index_5, index_6, index_7, logical_and_12, logical_and_13, logical_and_14, logical_and_15, logical_and_16, logical_and_17, logical_and_18, logical_and_19, logical_and_20, logical_and_21, logical_and_22, logical_and_23, lt_10, lt_11, lt_12, lt_13, lt_14, lt_15, lt_8, lt_9, mul_80, mul_81, mul_82, mul_83, mul_84, mul_85, mul_87, mul_88, mul_89, sub_15, sub_16, sub_17, sub_18, sub_19, sub_20, sub_21, sub_22, where_12, where_13, where_14, where_17, where_20, where_22, where_23
# Graph fragment:
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, 2.0), kwargs = {})
#   %add_33 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, 1.5), kwargs = {})
#   %floor : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_33,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, 2.0), kwargs = {})
#   %add_34 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, 1.5), kwargs = {})
#   %floor_1 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_34,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %lt_1), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt, %logical_and), kwargs = {})
#   %logical_and_2 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %logical_and_1), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %convert_element_type_2, %full_default_1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %convert_element_type_1, %full_default), kwargs = {})
#   %add_35 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor, 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_35, %add_33), kwargs = {})
#   %add_36 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_1, 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_36, %add_34), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %sub_4), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_2, %mul_37, %full_default_2), kwargs = {})
#   %ge_2 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_35, 0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_35, 4), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_1, 0), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_1, 4), kwargs = {})
#   %logical_and_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_3, %lt_3), kwargs = {})
#   %logical_and_4 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_2, %logical_and_3), kwargs = {})
#   %logical_and_5 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_2, %logical_and_4), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_4, [%view_10, %view_11, %where_4, %where_3]), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %floor), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_36, %add_34), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %sub_6), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_5, %mul_38, %full_default_5), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_1, %where_5), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor, 0), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor, 4), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_36, 0), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_36, 4), kwargs = {})
#   %logical_and_6 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_5, %lt_5), kwargs = {})
#   %logical_and_7 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_4, %logical_and_6), kwargs = {})
#   %logical_and_8 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_4, %logical_and_7), kwargs = {})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_4, [%view_10, %view_11, %where_7, %where_6]), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_35, %add_33), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %floor_1), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %sub_8), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_8, %mul_39, %full_default_8), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_2, %where_8), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_35, 0), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_35, 4), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_36, 0), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_36, 4), kwargs = {})
#   %logical_and_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_7, %lt_7), kwargs = {})
#   %logical_and_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_6, %logical_and_9), kwargs = {})
#   %logical_and_11 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_6, %logical_and_10), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_36, torch.int64), kwargs = {})
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %convert_element_type_8, %full_default_10), kwargs = {})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_4, [%view_10, %view_11, %where_10, %where_9]), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %floor), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %floor_1), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %sub_10), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_11, %mul_40, %full_default_11), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_3, %where_11), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_29, 2.0), kwargs = {})
#   %add_73 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, 1.5), kwargs = {})
#   %floor_2 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_73,), kwargs = {})
#   %ge_8 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_2, 0), kwargs = {})
#   %lt_8 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_2, 4), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_30, 2.0), kwargs = {})
#   %add_74 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, 1.5), kwargs = {})
#   %floor_3 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_74,), kwargs = {})
#   %ge_9 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_3, 0), kwargs = {})
#   %lt_9 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_3, 4), kwargs = {})
#   %logical_and_12 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_9, %lt_9), kwargs = {})
#   %logical_and_13 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_8, %logical_and_12), kwargs = {})
#   %logical_and_14 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_8, %logical_and_13), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_3, torch.int64), kwargs = {})
#   %full_default_13 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_14, %convert_element_type_11, %full_default_13), kwargs = {})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_2, torch.int64), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_14, %convert_element_type_10, %full_default_12), kwargs = {})
#   %add_75 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_2, 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_75, %add_73), kwargs = {})
#   %add_76 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_3, 1), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_76, %add_74), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %sub_16), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_14 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_14, %mul_82, %full_default_14), kwargs = {})
#   %ge_10 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_75, 0), kwargs = {})
#   %lt_10 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_75, 4), kwargs = {})
#   %ge_11 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_3, 0), kwargs = {})
#   %lt_11 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_3, 4), kwargs = {})
#   %logical_and_15 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_11, %lt_11), kwargs = {})
#   %logical_and_16 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_10, %logical_and_15), kwargs = {})
#   %logical_and_17 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_10, %logical_and_16), kwargs = {})
#   %index_5 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_29, [%view_35, %view_36, %where_16, %where_15]), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_73, %floor_2), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_76, %add_74), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %sub_18), kwargs = {})
#   %full_default_17 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_17, %mul_83, %full_default_17), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_5, %where_17), kwargs = {})
#   %ge_12 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_2, 0), kwargs = {})
#   %lt_12 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_2, 4), kwargs = {})
#   %ge_13 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_76, 0), kwargs = {})
#   %lt_13 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_76, 4), kwargs = {})
#   %logical_and_18 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_13, %lt_13), kwargs = {})
#   %logical_and_19 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_12, %logical_and_18), kwargs = {})
#   %logical_and_20 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_12, %logical_and_19), kwargs = {})
#   %index_6 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_29, [%view_35, %view_36, %where_19, %where_18]), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_75, %add_73), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_74, %floor_3), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %sub_20), kwargs = {})
#   %full_default_20 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_20 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_20, %mul_84, %full_default_20), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_6, %where_20), kwargs = {})
#   %ge_14 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_75, 0), kwargs = {})
#   %lt_14 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_75, 4), kwargs = {})
#   %ge_15 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_76, 0), kwargs = {})
#   %lt_15 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_76, 4), kwargs = {})
#   %logical_and_21 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_15, %lt_15), kwargs = {})
#   %logical_and_22 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_14, %logical_and_21), kwargs = {})
#   %logical_and_23 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_14, %logical_and_22), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_76, torch.int64), kwargs = {})
#   %full_default_22 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_23, %convert_element_type_17, %full_default_22), kwargs = {})
#   %index_7 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_29, [%view_35, %view_36, %where_22, %where_21]), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_73, %floor_2), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_74, %floor_3), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %sub_22), kwargs = {})
#   %full_default_23 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_23 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_23, %mul_85, %full_default_23), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_7, %where_23), kwargs = {})
triton_poi_fused_grid_sampler_2d_2 = async_compile.triton('triton_poi_fused_grid_sampler_2d_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'out_ptr3': '*i64', 'out_ptr4': '*i64', 'out_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_2(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x4 = xindex // 16
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 32*x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (16 + x0 + 32*x2), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr1 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.3333333333333333
    tmp7 = tmp5 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp11 * tmp4
    tmp13 = 1.5
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 0.0
    tmp17 = tmp15 >= tmp16
    tmp18 = 4.0
    tmp19 = tmp15 < tmp18
    tmp20 = tmp1 == tmp1
    tmp21 = tl.where(tmp20, tmp9, tmp3)
    tmp22 = tmp21 * tmp4
    tmp23 = tmp22 + tmp13
    tmp24 = libdevice.floor(tmp23)
    tmp25 = tmp24 >= tmp16
    tmp26 = tmp24 < tmp18
    tmp27 = tmp25 & tmp26
    tmp28 = tmp19 & tmp27
    tmp29 = tmp17 & tmp28
    tmp30 = tmp24.to(tl.int64)
    tmp31 = tl.full([1], 0, tl.int64)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tmp15.to(tl.int64)
    tmp34 = tl.where(tmp29, tmp33, tmp31)
    tmp35 = tmp15 + tmp8
    tmp36 = tmp35 - tmp14
    tmp37 = tmp24 + tmp8
    tmp38 = tmp37 - tmp23
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp29, tmp39, tmp16)
    tmp42 = tmp41 * tmp4
    tmp43 = tmp42 * tmp6
    tmp44 = tmp43 - tmp8
    tmp46 = tl.where(tmp2, tmp44, tmp45)
    tmp47 = tmp46 * tmp4
    tmp48 = tmp47 + tmp13
    tmp49 = libdevice.floor(tmp48)
    tmp50 = tmp49 >= tmp16
    tmp51 = tmp49 < tmp18
    tmp52 = tl.where(tmp20, tmp44, tmp41)
    tmp53 = tmp52 * tmp4
    tmp54 = tmp53 + tmp13
    tmp55 = libdevice.floor(tmp54)
    tmp56 = tmp55 >= tmp16
    tmp57 = tmp55 < tmp18
    tmp58 = tmp56 & tmp57
    tmp59 = tmp51 & tmp58
    tmp60 = tmp50 & tmp59
    tmp61 = tmp55.to(tl.int64)
    tmp62 = tl.where(tmp60, tmp61, tmp31)
    tmp63 = tmp49.to(tl.int64)
    tmp64 = tl.where(tmp60, tmp63, tmp31)
    tmp65 = tmp49 + tmp8
    tmp66 = tmp65 - tmp48
    tmp67 = tmp55 + tmp8
    tmp68 = tmp67 - tmp54
    tmp69 = tmp66 * tmp68
    tmp70 = tl.where(tmp60, tmp69, tmp16)
    tmp71 = tmp35 < tmp18
    tmp72 = tmp37 >= tmp16
    tmp73 = tmp37 < tmp18
    tmp74 = tmp72 & tmp73
    tmp75 = tmp71 & tmp74
    tmp76 = tmp19 & tmp74
    tmp77 = tmp17 & tmp76
    tmp78 = tmp35 >= tmp16
    tmp79 = tmp71 & tmp27
    tmp80 = tmp78 & tmp79
    tmp81 = tmp65 < tmp18
    tmp82 = tmp67 >= tmp16
    tmp83 = tmp67 < tmp18
    tmp84 = tmp82 & tmp83
    tmp85 = tmp81 & tmp84
    tmp86 = tmp51 & tmp84
    tmp87 = tmp50 & tmp86
    tmp88 = tmp65 >= tmp16
    tmp89 = tmp81 & tmp58
    tmp90 = tmp88 & tmp89
    tmp91 = tl.where(tmp90, tmp61, tmp31)
    tmp92 = tl.full([XBLOCK], 4, tl.int32)
    tmp93 = tmp91 + tmp92
    tmp94 = tmp91 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp91)
    tl.device_assert((0 <= tmp95) & (tmp95 < 4), "index out of bounds: 0 <= tmp95 < 4")
    tmp97 = tmp65.to(tl.int64)
    tmp98 = tl.where(tmp90, tmp97, tmp31)
    tmp99 = tmp98 + tmp92
    tmp100 = tmp98 < 0
    tmp101 = tl.where(tmp100, tmp99, tmp98)
    tl.device_assert((0 <= tmp101) & (tmp101 < 4), "index out of bounds: 0 <= tmp101 < 4")
    tmp103 = tl.load(in_ptr2 + (tmp101 + 4*tmp95 + 16*x4), None, eviction_policy='evict_last')
    tmp104 = tmp48 - tmp49
    tmp105 = tmp104 * tmp68
    tmp106 = tl.where(tmp90, tmp105, tmp16)
    tmp107 = tmp103 * tmp106
    tmp108 = tmp67.to(tl.int64)
    tmp109 = tl.where(tmp87, tmp108, tmp31)
    tmp110 = tmp109 + tmp92
    tmp111 = tmp109 < 0
    tmp112 = tl.where(tmp111, tmp110, tmp109)
    tl.device_assert((0 <= tmp112) & (tmp112 < 4), "index out of bounds: 0 <= tmp112 < 4")
    tmp114 = tl.where(tmp87, tmp63, tmp31)
    tmp115 = tmp114 + tmp92
    tmp116 = tmp114 < 0
    tmp117 = tl.where(tmp116, tmp115, tmp114)
    tl.device_assert((0 <= tmp117) & (tmp117 < 4), "index out of bounds: 0 <= tmp117 < 4")
    tmp119 = tl.load(in_ptr2 + (tmp117 + 4*tmp112 + 16*x4), None, eviction_policy='evict_last')
    tmp120 = tmp54 - tmp55
    tmp121 = tmp66 * tmp120
    tmp122 = tl.where(tmp87, tmp121, tmp16)
    tmp123 = tmp119 * tmp122
    tmp124 = tmp88 & tmp85
    tmp125 = tmp104 * tmp120
    tmp126 = tl.where(tmp124, tmp125, tmp16)
    tmp127 = tl.where(tmp124, tmp108, tmp31)
    tmp128 = tmp127 + tmp92
    tmp129 = tmp127 < 0
    tmp130 = tl.where(tmp129, tmp128, tmp127)
    tl.device_assert((0 <= tmp130) & (tmp130 < 4), "index out of bounds: 0 <= tmp130 < 4")
    tmp132 = tl.where(tmp124, tmp97, tmp31)
    tmp133 = tmp132 + tmp92
    tmp134 = tmp132 < 0
    tmp135 = tl.where(tmp134, tmp133, tmp132)
    tl.device_assert((0 <= tmp135) & (tmp135 < 4), "index out of bounds: 0 <= tmp135 < 4")
    tmp137 = tl.load(in_ptr2 + (tmp135 + 4*tmp130 + 16*x4), None, eviction_policy='evict_last')
    tmp138 = tmp137 * tmp126
    tmp139 = tl.where(tmp80, tmp30, tmp31)
    tmp140 = tmp139 + tmp92
    tmp141 = tmp139 < 0
    tmp142 = tl.where(tmp141, tmp140, tmp139)
    tl.device_assert((0 <= tmp142) & (tmp142 < 4), "index out of bounds: 0 <= tmp142 < 4")
    tmp144 = tmp35.to(tl.int64)
    tmp145 = tl.where(tmp80, tmp144, tmp31)
    tmp146 = tmp145 + tmp92
    tmp147 = tmp145 < 0
    tmp148 = tl.where(tmp147, tmp146, tmp145)
    tl.device_assert((0 <= tmp148) & (tmp148 < 4), "index out of bounds: 0 <= tmp148 < 4")
    tmp150 = tl.load(in_ptr2 + (tmp148 + 4*tmp142 + 16*x4), None, eviction_policy='evict_last')
    tmp151 = tmp14 - tmp15
    tmp152 = tmp151 * tmp38
    tmp153 = tl.where(tmp80, tmp152, tmp16)
    tmp154 = tmp150 * tmp153
    tmp155 = tmp37.to(tl.int64)
    tmp156 = tl.where(tmp77, tmp155, tmp31)
    tmp157 = tmp156 + tmp92
    tmp158 = tmp156 < 0
    tmp159 = tl.where(tmp158, tmp157, tmp156)
    tl.device_assert((0 <= tmp159) & (tmp159 < 4), "index out of bounds: 0 <= tmp159 < 4")
    tmp161 = tl.where(tmp77, tmp33, tmp31)
    tmp162 = tmp161 + tmp92
    tmp163 = tmp161 < 0
    tmp164 = tl.where(tmp163, tmp162, tmp161)
    tl.device_assert((0 <= tmp164) & (tmp164 < 4), "index out of bounds: 0 <= tmp164 < 4")
    tmp166 = tl.load(in_ptr2 + (tmp164 + 4*tmp159 + 16*x4), None, eviction_policy='evict_last')
    tmp167 = tmp23 - tmp24
    tmp168 = tmp36 * tmp167
    tmp169 = tl.where(tmp77, tmp168, tmp16)
    tmp170 = tmp166 * tmp169
    tmp171 = tmp78 & tmp75
    tmp172 = tmp151 * tmp167
    tmp173 = tl.where(tmp171, tmp172, tmp16)
    tmp174 = tl.where(tmp171, tmp155, tmp31)
    tmp175 = tmp174 + tmp92
    tmp176 = tmp174 < 0
    tmp177 = tl.where(tmp176, tmp175, tmp174)
    tl.device_assert((0 <= tmp177) & (tmp177 < 4), "index out of bounds: 0 <= tmp177 < 4")
    tmp179 = tl.where(tmp171, tmp144, tmp31)
    tmp180 = tmp179 + tmp92
    tmp181 = tmp179 < 0
    tmp182 = tl.where(tmp181, tmp180, tmp179)
    tl.device_assert((0 <= tmp182) & (tmp182 < 4), "index out of bounds: 0 <= tmp182 < 4")
    tmp184 = tl.load(in_ptr2 + (tmp182 + 4*tmp177 + 16*x4), None, eviction_policy='evict_last')
    tmp185 = tmp184 * tmp173
    tl.store(out_ptr0 + (x3), tmp32, None)
    tl.store(out_ptr1 + (x3), tmp34, None)
    tl.store(out_ptr2 + (x3), tmp40, None)
    tl.store(out_ptr3 + (x3), tmp62, None)
    tl.store(out_ptr4 + (x3), tmp64, None)
    tl.store(out_ptr5 + (x3), tmp70, None)
    tl.store(in_out_ptr0 + (x3), tmp107, None)
    tl.store(in_out_ptr1 + (x3), tmp123, None)
    tl.store(in_out_ptr2 + (x3), tmp138, None)
    tl.store(in_out_ptr3 + (x3), tmp154, None)
    tl.store(in_out_ptr4 + (x3), tmp170, None)
    tl.store(in_out_ptr5 + (x3), tmp185, None)
''', device_str='cuda')


# kernel path: inductor_cache/mg/cmggrrefyyngtfixenrh6wsqek6eshrmo5aik2i7gbeonyg6woi6.py
# Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.grid_sampler_2d]
# Source node to ATen node mapping:
#   output_2 => add_113, add_114, add_115, add_116, convert_element_type_19, convert_element_type_20, convert_element_type_26, floor_4, floor_5, full_default_24, full_default_25, full_default_26, full_default_29, full_default_32, full_default_34, full_default_35, ge_16, ge_17, ge_18, ge_19, ge_20, ge_21, ge_22, ge_23, index_10, index_11, index_9, logical_and_24, logical_and_25, logical_and_26, logical_and_27, logical_and_28, logical_and_29, logical_and_30, logical_and_31, logical_and_32, logical_and_33, logical_and_34, logical_and_35, lt_16, lt_17, lt_18, lt_19, lt_20, lt_21, lt_22, lt_23, mul_125, mul_126, mul_127, mul_128, mul_129, mul_130, mul_132, mul_133, mul_134, sub_27, sub_28, sub_29, sub_30, sub_31, sub_32, sub_33, sub_34, where_24, where_25, where_26, where_29, where_32, where_34, where_35
# Graph fragment:
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_43, 2.0), kwargs = {})
#   %add_113 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, 1.5), kwargs = {})
#   %floor_4 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_113,), kwargs = {})
#   %ge_16 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_4, 0), kwargs = {})
#   %lt_16 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_4, 4), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_44, 2.0), kwargs = {})
#   %add_114 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_126, 1.5), kwargs = {})
#   %floor_5 : [num_users=9] = call_function[target=torch.ops.aten.floor.default](args = (%add_114,), kwargs = {})
#   %ge_17 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_5, 0), kwargs = {})
#   %lt_17 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_5, 4), kwargs = {})
#   %logical_and_24 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_17, %lt_17), kwargs = {})
#   %logical_and_25 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_16, %logical_and_24), kwargs = {})
#   %logical_and_26 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_16, %logical_and_25), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_5, torch.int64), kwargs = {})
#   %full_default_25 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_25 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_26, %convert_element_type_20, %full_default_25), kwargs = {})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_4, torch.int64), kwargs = {})
#   %full_default_24 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_26, %convert_element_type_19, %full_default_24), kwargs = {})
#   %add_115 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_4, 1), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_115, %add_113), kwargs = {})
#   %add_116 : [num_users=8] = call_function[target=torch.ops.aten.add.Tensor](args = (%floor_5, 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_116, %add_114), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %sub_28), kwargs = {})
#   %full_default_26 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_26, %mul_127, %full_default_26), kwargs = {})
#   %ge_18 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_115, 0), kwargs = {})
#   %lt_18 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_115, 4), kwargs = {})
#   %ge_19 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_5, 0), kwargs = {})
#   %lt_19 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_5, 4), kwargs = {})
#   %logical_and_27 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_19, %lt_19), kwargs = {})
#   %logical_and_28 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_18, %logical_and_27), kwargs = {})
#   %logical_and_29 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_18, %logical_and_28), kwargs = {})
#   %index_9 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_54, [%view_60, %view_61, %where_28, %where_27]), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_113, %floor_4), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_116, %add_114), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %sub_30), kwargs = {})
#   %full_default_29 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_29 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_29, %mul_128, %full_default_29), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_9, %where_29), kwargs = {})
#   %ge_20 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%floor_4, 0), kwargs = {})
#   %lt_20 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%floor_4, 4), kwargs = {})
#   %ge_21 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_116, 0), kwargs = {})
#   %lt_21 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_116, 4), kwargs = {})
#   %logical_and_30 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_21, %lt_21), kwargs = {})
#   %logical_and_31 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_20, %logical_and_30), kwargs = {})
#   %logical_and_32 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_20, %logical_and_31), kwargs = {})
#   %index_10 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_54, [%view_60, %view_61, %where_31, %where_30]), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_115, %add_113), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_114, %floor_5), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %sub_32), kwargs = {})
#   %full_default_32 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_32 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_32, %mul_129, %full_default_32), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_10, %where_32), kwargs = {})
#   %ge_22 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_115, 0), kwargs = {})
#   %lt_22 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_115, 4), kwargs = {})
#   %ge_23 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_116, 0), kwargs = {})
#   %lt_23 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_116, 4), kwargs = {})
#   %logical_and_33 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_23, %lt_23), kwargs = {})
#   %logical_and_34 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%lt_22, %logical_and_33), kwargs = {})
#   %logical_and_35 : [num_users=3] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_22, %logical_and_34), kwargs = {})
#   %convert_element_type_26 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_116, torch.int64), kwargs = {})
#   %full_default_34 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_34 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_35, %convert_element_type_26, %full_default_34), kwargs = {})
#   %index_11 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_54, [%view_60, %view_61, %where_34, %where_33]), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_113, %floor_4), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_114, %floor_5), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %sub_34), kwargs = {})
#   %full_default_35 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_35 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_35, %mul_130, %full_default_35), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_11, %where_35), kwargs = {})
triton_poi_fused_grid_sampler_2d_3 = async_compile.triton('triton_poi_fused_grid_sampler_2d_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_3(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x4 = xindex // 16
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 32*x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.3333333333333333
    tmp7 = tmp5 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp11 * tmp4
    tmp13 = 1.5
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 0.0
    tmp17 = tmp15 >= tmp16
    tmp18 = 4.0
    tmp19 = tmp15 < tmp18
    tmp20 = tmp1 == tmp1
    tmp21 = tl.where(tmp20, tmp9, tmp3)
    tmp22 = tmp21 * tmp4
    tmp23 = tmp22 + tmp13
    tmp24 = libdevice.floor(tmp23)
    tmp25 = tmp24 >= tmp16
    tmp26 = tmp24 < tmp18
    tmp27 = tmp25 & tmp26
    tmp28 = tmp19 & tmp27
    tmp29 = tmp17 & tmp28
    tmp30 = tmp24.to(tl.int64)
    tmp31 = tl.full([1], 0, tl.int64)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tmp15.to(tl.int64)
    tmp34 = tl.where(tmp29, tmp33, tmp31)
    tmp35 = tmp15 + tmp8
    tmp36 = tmp35 - tmp14
    tmp37 = tmp24 + tmp8
    tmp38 = tmp37 - tmp23
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp29, tmp39, tmp16)
    tmp41 = tmp35 < tmp18
    tmp42 = tmp37 >= tmp16
    tmp43 = tmp37 < tmp18
    tmp44 = tmp42 & tmp43
    tmp45 = tmp41 & tmp44
    tmp46 = tmp19 & tmp44
    tmp47 = tmp17 & tmp46
    tmp48 = tmp35 >= tmp16
    tmp49 = tmp41 & tmp27
    tmp50 = tmp48 & tmp49
    tmp51 = tl.where(tmp50, tmp30, tmp31)
    tmp52 = tl.full([XBLOCK], 4, tl.int32)
    tmp53 = tmp51 + tmp52
    tmp54 = tmp51 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp51)
    tl.device_assert((0 <= tmp55) & (tmp55 < 4), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tmp35.to(tl.int64)
    tmp58 = tl.where(tmp50, tmp57, tmp31)
    tmp59 = tmp58 + tmp52
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tl.device_assert((0 <= tmp61) & (tmp61 < 4), "index out of bounds: 0 <= tmp61 < 4")
    tmp63 = tl.load(in_ptr1 + (tmp61 + 4*tmp55 + 16*x4), None, eviction_policy='evict_last')
    tmp64 = tmp14 - tmp15
    tmp65 = tmp64 * tmp38
    tmp66 = tl.where(tmp50, tmp65, tmp16)
    tmp67 = tmp63 * tmp66
    tmp68 = tmp37.to(tl.int64)
    tmp69 = tl.where(tmp47, tmp68, tmp31)
    tmp70 = tmp69 + tmp52
    tmp71 = tmp69 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp69)
    tl.device_assert((0 <= tmp72) & (tmp72 < 4), "index out of bounds: 0 <= tmp72 < 4")
    tmp74 = tl.where(tmp47, tmp33, tmp31)
    tmp75 = tmp74 + tmp52
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tl.device_assert((0 <= tmp77) & (tmp77 < 4), "index out of bounds: 0 <= tmp77 < 4")
    tmp79 = tl.load(in_ptr1 + (tmp77 + 4*tmp72 + 16*x4), None, eviction_policy='evict_last')
    tmp80 = tmp23 - tmp24
    tmp81 = tmp36 * tmp80
    tmp82 = tl.where(tmp47, tmp81, tmp16)
    tmp83 = tmp79 * tmp82
    tmp84 = tmp48 & tmp45
    tmp85 = tmp64 * tmp80
    tmp86 = tl.where(tmp84, tmp85, tmp16)
    tmp87 = tl.where(tmp84, tmp68, tmp31)
    tmp88 = tmp87 + tmp52
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tl.device_assert((0 <= tmp90) & (tmp90 < 4), "index out of bounds: 0 <= tmp90 < 4")
    tmp92 = tl.where(tmp84, tmp57, tmp31)
    tmp93 = tmp92 + tmp52
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert((0 <= tmp95) & (tmp95 < 4), "index out of bounds: 0 <= tmp95 < 4")
    tmp97 = tl.load(in_ptr1 + (tmp95 + 4*tmp90 + 16*x4), None, eviction_policy='evict_last')
    tmp98 = tmp97 * tmp86
    tl.store(out_ptr0 + (x3), tmp32, None)
    tl.store(out_ptr1 + (x3), tmp34, None)
    tl.store(out_ptr2 + (x3), tmp40, None)
    tl.store(in_out_ptr0 + (x3), tmp67, None)
    tl.store(in_out_ptr1 + (x3), tmp83, None)
    tl.store(in_out_ptr2 + (x3), tmp98, None)
''', device_str='cuda')


# kernel path: inductor_cache/rw/crwqui53wpq2b2e3yuou7udu5imdn7n2ortmuxg4iesh42oboror.py
# Topologically Sorted Source Nodes: [output, sub_3, norm, output_1, sub_7, norm_1, output_2, sub_11, norm_2], Original ATen: [aten.grid_sampler_2d, aten.sub, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   norm => abs_1, sum_1
#   norm_1 => abs_2, sum_2
#   norm_2 => abs_3, sum_3
#   output => add_37, add_38, add_39, index, mul_41
#   output_1 => add_77, add_78, add_79, index_4, mul_86
#   output_2 => add_117, add_118, add_119, index_8, mul_131
#   sub_11 => sub_35
#   sub_3 => sub_11
#   sub_7 => sub_23
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_4, [%view_10, %view_11, %where_1, %where]), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %where_2), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %mul_42), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %mul_43), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %mul_44), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %add_39), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_11,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [1]), kwargs = {})
#   %index_4 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_29, [%view_35, %view_36, %where_13, %where_12]), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_4, %where_14), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %mul_87), kwargs = {})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_77, %mul_88), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %mul_89), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_28, %add_79), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_23,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_2, [1]), kwargs = {})
#   %index_8 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_54, [%view_60, %view_61, %where_25, %where_24]), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_8, %where_26), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %mul_132), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %mul_133), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %mul_134), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_53, %add_119), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_35,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_3, [1]), kwargs = {})
triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_4 = async_compile.triton('triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*i64', 'in_ptr15': '*i64', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 76, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr2 + (x0 + 64*x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr5 + (x0 + 64*x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x0 + 64*x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x0 + 64*x1), xmask)
    tmp23 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp24 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp29 = tl.load(in_ptr2 + (16 + x0 + 64*x1), xmask)
    tmp35 = tl.load(in_ptr4 + (16 + x0 + 64*x1), xmask)
    tmp37 = tl.load(in_ptr5 + (16 + x0 + 64*x1), xmask)
    tmp39 = tl.load(in_ptr6 + (16 + x0 + 64*x1), xmask)
    tmp41 = tl.load(in_ptr7 + (16 + x0 + 64*x1), xmask)
    tmp46 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp47 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp52 = tl.load(in_ptr2 + (32 + x0 + 64*x1), xmask)
    tmp58 = tl.load(in_ptr4 + (32 + x0 + 64*x1), xmask)
    tmp60 = tl.load(in_ptr5 + (32 + x0 + 64*x1), xmask)
    tmp62 = tl.load(in_ptr6 + (32 + x0 + 64*x1), xmask)
    tmp64 = tl.load(in_ptr7 + (32 + x0 + 64*x1), xmask)
    tmp69 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp70 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp75 = tl.load(in_ptr2 + (48 + x0 + 64*x1), xmask)
    tmp81 = tl.load(in_ptr4 + (48 + x0 + 64*x1), xmask)
    tmp83 = tl.load(in_ptr5 + (48 + x0 + 64*x1), xmask)
    tmp85 = tl.load(in_ptr6 + (48 + x0 + 64*x1), xmask)
    tmp87 = tl.load(in_ptr7 + (48 + x0 + 64*x1), xmask)
    tmp92 = tl.load(in_ptr8 + (x0 + 64*x1), xmask)
    tmp97 = tl.load(in_ptr9 + (x0 + 64*x1), xmask)
    tmp103 = tl.load(in_ptr10 + (x0 + 64*x1), xmask)
    tmp105 = tl.load(in_ptr11 + (x0 + 64*x1), xmask)
    tmp107 = tl.load(in_ptr12 + (x0 + 64*x1), xmask)
    tmp109 = tl.load(in_ptr13 + (x0 + 64*x1), xmask)
    tmp113 = tl.load(in_ptr8 + (16 + x0 + 64*x1), xmask)
    tmp118 = tl.load(in_ptr9 + (16 + x0 + 64*x1), xmask)
    tmp124 = tl.load(in_ptr10 + (16 + x0 + 64*x1), xmask)
    tmp126 = tl.load(in_ptr11 + (16 + x0 + 64*x1), xmask)
    tmp128 = tl.load(in_ptr12 + (16 + x0 + 64*x1), xmask)
    tmp130 = tl.load(in_ptr13 + (16 + x0 + 64*x1), xmask)
    tmp135 = tl.load(in_ptr8 + (32 + x0 + 64*x1), xmask)
    tmp140 = tl.load(in_ptr9 + (32 + x0 + 64*x1), xmask)
    tmp146 = tl.load(in_ptr10 + (32 + x0 + 64*x1), xmask)
    tmp148 = tl.load(in_ptr11 + (32 + x0 + 64*x1), xmask)
    tmp150 = tl.load(in_ptr12 + (32 + x0 + 64*x1), xmask)
    tmp152 = tl.load(in_ptr13 + (32 + x0 + 64*x1), xmask)
    tmp157 = tl.load(in_ptr8 + (48 + x0 + 64*x1), xmask)
    tmp162 = tl.load(in_ptr9 + (48 + x0 + 64*x1), xmask)
    tmp168 = tl.load(in_ptr10 + (48 + x0 + 64*x1), xmask)
    tmp170 = tl.load(in_ptr11 + (48 + x0 + 64*x1), xmask)
    tmp172 = tl.load(in_ptr12 + (48 + x0 + 64*x1), xmask)
    tmp174 = tl.load(in_ptr13 + (48 + x0 + 64*x1), xmask)
    tmp179 = tl.load(in_ptr14 + (x0 + 64*x1), xmask)
    tmp184 = tl.load(in_ptr15 + (x0 + 64*x1), xmask)
    tmp190 = tl.load(in_ptr16 + (x0 + 64*x1), xmask)
    tmp192 = tl.load(in_ptr17 + (x0 + 64*x1), xmask)
    tmp194 = tl.load(in_ptr18 + (x0 + 64*x1), xmask)
    tmp196 = tl.load(in_ptr19 + (x0 + 64*x1), xmask)
    tmp200 = tl.load(in_ptr14 + (16 + x0 + 64*x1), xmask)
    tmp205 = tl.load(in_ptr15 + (16 + x0 + 64*x1), xmask)
    tmp211 = tl.load(in_ptr16 + (16 + x0 + 64*x1), xmask)
    tmp213 = tl.load(in_ptr17 + (16 + x0 + 64*x1), xmask)
    tmp215 = tl.load(in_ptr18 + (16 + x0 + 64*x1), xmask)
    tmp217 = tl.load(in_ptr19 + (16 + x0 + 64*x1), xmask)
    tmp222 = tl.load(in_ptr14 + (32 + x0 + 64*x1), xmask)
    tmp227 = tl.load(in_ptr15 + (32 + x0 + 64*x1), xmask)
    tmp233 = tl.load(in_ptr16 + (32 + x0 + 64*x1), xmask)
    tmp235 = tl.load(in_ptr17 + (32 + x0 + 64*x1), xmask)
    tmp237 = tl.load(in_ptr18 + (32 + x0 + 64*x1), xmask)
    tmp239 = tl.load(in_ptr19 + (32 + x0 + 64*x1), xmask)
    tmp244 = tl.load(in_ptr14 + (48 + x0 + 64*x1), xmask)
    tmp249 = tl.load(in_ptr15 + (48 + x0 + 64*x1), xmask)
    tmp255 = tl.load(in_ptr16 + (48 + x0 + 64*x1), xmask)
    tmp257 = tl.load(in_ptr17 + (48 + x0 + 64*x1), xmask)
    tmp259 = tl.load(in_ptr18 + (48 + x0 + 64*x1), xmask)
    tmp261 = tl.load(in_ptr19 + (48 + x0 + 64*x1), xmask)
    tmp2 = tl.full([XBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 4")
    tmp8 = tmp7 + tmp2
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 4")
    tmp12 = tl.load(in_ptr3 + (tmp10 + 4*tmp5 + 64*x1), xmask, eviction_policy='evict_last')
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tmp0 - tmp20
    tmp22 = tl_math.abs(tmp21)
    tmp25 = tmp24 + tmp2
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 4")
    tmp30 = tmp29 + tmp2
    tmp31 = tmp29 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp29)
    tl.device_assert(((0 <= tmp32) & (tmp32 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp32 < 4")
    tmp34 = tl.load(in_ptr3 + (16 + tmp32 + 4*tmp27 + 64*x1), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp23 - tmp42
    tmp44 = tl_math.abs(tmp43)
    tmp45 = tmp22 + tmp44
    tmp48 = tmp47 + tmp2
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tl.device_assert(((0 <= tmp50) & (tmp50 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp50 < 4")
    tmp53 = tmp52 + tmp2
    tmp54 = tmp52 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp52)
    tl.device_assert(((0 <= tmp55) & (tmp55 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tl.load(in_ptr3 + (32 + tmp55 + 4*tmp50 + 64*x1), xmask, eviction_policy='evict_last')
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp63 = tmp61 + tmp62
    tmp65 = tmp63 + tmp64
    tmp66 = tmp46 - tmp65
    tmp67 = tl_math.abs(tmp66)
    tmp68 = tmp45 + tmp67
    tmp71 = tmp70 + tmp2
    tmp72 = tmp70 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp70)
    tl.device_assert(((0 <= tmp73) & (tmp73 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp73 < 4")
    tmp76 = tmp75 + tmp2
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tl.device_assert(((0 <= tmp78) & (tmp78 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp78 < 4")
    tmp80 = tl.load(in_ptr3 + (48 + tmp78 + 4*tmp73 + 64*x1), xmask, eviction_policy='evict_last')
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp86 = tmp84 + tmp85
    tmp88 = tmp86 + tmp87
    tmp89 = tmp69 - tmp88
    tmp90 = tl_math.abs(tmp89)
    tmp91 = tmp68 + tmp90
    tmp93 = tmp92 + tmp2
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert(((0 <= tmp95) & (tmp95 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp95 < 4")
    tmp98 = tmp97 + tmp2
    tmp99 = tmp97 < 0
    tmp100 = tl.where(tmp99, tmp98, tmp97)
    tl.device_assert(((0 <= tmp100) & (tmp100 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp100 < 4")
    tmp102 = tl.load(in_ptr3 + (tmp100 + 4*tmp95 + 64*x1), xmask, eviction_policy='evict_last')
    tmp104 = tmp102 * tmp103
    tmp106 = tmp104 + tmp105
    tmp108 = tmp106 + tmp107
    tmp110 = tmp108 + tmp109
    tmp111 = tmp0 - tmp110
    tmp112 = tl_math.abs(tmp111)
    tmp114 = tmp113 + tmp2
    tmp115 = tmp113 < 0
    tmp116 = tl.where(tmp115, tmp114, tmp113)
    tl.device_assert(((0 <= tmp116) & (tmp116 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp116 < 4")
    tmp119 = tmp118 + tmp2
    tmp120 = tmp118 < 0
    tmp121 = tl.where(tmp120, tmp119, tmp118)
    tl.device_assert(((0 <= tmp121) & (tmp121 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp121 < 4")
    tmp123 = tl.load(in_ptr3 + (16 + tmp121 + 4*tmp116 + 64*x1), xmask, eviction_policy='evict_last')
    tmp125 = tmp123 * tmp124
    tmp127 = tmp125 + tmp126
    tmp129 = tmp127 + tmp128
    tmp131 = tmp129 + tmp130
    tmp132 = tmp23 - tmp131
    tmp133 = tl_math.abs(tmp132)
    tmp134 = tmp112 + tmp133
    tmp136 = tmp135 + tmp2
    tmp137 = tmp135 < 0
    tmp138 = tl.where(tmp137, tmp136, tmp135)
    tl.device_assert(((0 <= tmp138) & (tmp138 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp138 < 4")
    tmp141 = tmp140 + tmp2
    tmp142 = tmp140 < 0
    tmp143 = tl.where(tmp142, tmp141, tmp140)
    tl.device_assert(((0 <= tmp143) & (tmp143 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp143 < 4")
    tmp145 = tl.load(in_ptr3 + (32 + tmp143 + 4*tmp138 + 64*x1), xmask, eviction_policy='evict_last')
    tmp147 = tmp145 * tmp146
    tmp149 = tmp147 + tmp148
    tmp151 = tmp149 + tmp150
    tmp153 = tmp151 + tmp152
    tmp154 = tmp46 - tmp153
    tmp155 = tl_math.abs(tmp154)
    tmp156 = tmp134 + tmp155
    tmp158 = tmp157 + tmp2
    tmp159 = tmp157 < 0
    tmp160 = tl.where(tmp159, tmp158, tmp157)
    tl.device_assert(((0 <= tmp160) & (tmp160 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp160 < 4")
    tmp163 = tmp162 + tmp2
    tmp164 = tmp162 < 0
    tmp165 = tl.where(tmp164, tmp163, tmp162)
    tl.device_assert(((0 <= tmp165) & (tmp165 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp165 < 4")
    tmp167 = tl.load(in_ptr3 + (48 + tmp165 + 4*tmp160 + 64*x1), xmask, eviction_policy='evict_last')
    tmp169 = tmp167 * tmp168
    tmp171 = tmp169 + tmp170
    tmp173 = tmp171 + tmp172
    tmp175 = tmp173 + tmp174
    tmp176 = tmp69 - tmp175
    tmp177 = tl_math.abs(tmp176)
    tmp178 = tmp156 + tmp177
    tmp180 = tmp179 + tmp2
    tmp181 = tmp179 < 0
    tmp182 = tl.where(tmp181, tmp180, tmp179)
    tl.device_assert(((0 <= tmp182) & (tmp182 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp182 < 4")
    tmp185 = tmp184 + tmp2
    tmp186 = tmp184 < 0
    tmp187 = tl.where(tmp186, tmp185, tmp184)
    tl.device_assert(((0 <= tmp187) & (tmp187 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp187 < 4")
    tmp189 = tl.load(in_ptr3 + (tmp187 + 4*tmp182 + 64*x1), xmask, eviction_policy='evict_last')
    tmp191 = tmp189 * tmp190
    tmp193 = tmp191 + tmp192
    tmp195 = tmp193 + tmp194
    tmp197 = tmp195 + tmp196
    tmp198 = tmp0 - tmp197
    tmp199 = tl_math.abs(tmp198)
    tmp201 = tmp200 + tmp2
    tmp202 = tmp200 < 0
    tmp203 = tl.where(tmp202, tmp201, tmp200)
    tl.device_assert(((0 <= tmp203) & (tmp203 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp203 < 4")
    tmp206 = tmp205 + tmp2
    tmp207 = tmp205 < 0
    tmp208 = tl.where(tmp207, tmp206, tmp205)
    tl.device_assert(((0 <= tmp208) & (tmp208 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp208 < 4")
    tmp210 = tl.load(in_ptr3 + (16 + tmp208 + 4*tmp203 + 64*x1), xmask, eviction_policy='evict_last')
    tmp212 = tmp210 * tmp211
    tmp214 = tmp212 + tmp213
    tmp216 = tmp214 + tmp215
    tmp218 = tmp216 + tmp217
    tmp219 = tmp23 - tmp218
    tmp220 = tl_math.abs(tmp219)
    tmp221 = tmp199 + tmp220
    tmp223 = tmp222 + tmp2
    tmp224 = tmp222 < 0
    tmp225 = tl.where(tmp224, tmp223, tmp222)
    tl.device_assert(((0 <= tmp225) & (tmp225 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp225 < 4")
    tmp228 = tmp227 + tmp2
    tmp229 = tmp227 < 0
    tmp230 = tl.where(tmp229, tmp228, tmp227)
    tl.device_assert(((0 <= tmp230) & (tmp230 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp230 < 4")
    tmp232 = tl.load(in_ptr3 + (32 + tmp230 + 4*tmp225 + 64*x1), xmask, eviction_policy='evict_last')
    tmp234 = tmp232 * tmp233
    tmp236 = tmp234 + tmp235
    tmp238 = tmp236 + tmp237
    tmp240 = tmp238 + tmp239
    tmp241 = tmp46 - tmp240
    tmp242 = tl_math.abs(tmp241)
    tmp243 = tmp221 + tmp242
    tmp245 = tmp244 + tmp2
    tmp246 = tmp244 < 0
    tmp247 = tl.where(tmp246, tmp245, tmp244)
    tl.device_assert(((0 <= tmp247) & (tmp247 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp247 < 4")
    tmp250 = tmp249 + tmp2
    tmp251 = tmp249 < 0
    tmp252 = tl.where(tmp251, tmp250, tmp249)
    tl.device_assert(((0 <= tmp252) & (tmp252 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp252 < 4")
    tmp254 = tl.load(in_ptr3 + (48 + tmp252 + 4*tmp247 + 64*x1), xmask, eviction_policy='evict_last')
    tmp256 = tmp254 * tmp255
    tmp258 = tmp256 + tmp257
    tmp260 = tmp258 + tmp259
    tmp262 = tmp260 + tmp261
    tmp263 = tmp69 - tmp262
    tmp264 = tl_math.abs(tmp263)
    tmp265 = tmp243 + tmp264
    tl.store(out_ptr0 + (x2), tmp91, xmask)
    tl.store(out_ptr1 + (x2), tmp178, xmask)
    tl.store(out_ptr2 + (x2), tmp265, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pi/cpikqq5wfegyubfrm7664ccc32z7wij5vg35sikcohrbvkxujgit.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/43/c437quvjf3dq2qvqpsadwn5d26lxpfdqb5gqw5ltrzzwzi6sflik.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y6/cy66nlpq2f7rpuqw5tljxnk3bujnx33n4g6ejbctwfwgtkbxqlvm.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/he/cheppyzvdso2lilsuujcnsfew5kgfzaalntlezzj3loyclx5zilc.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gb/cgbt5k4jjtfkb7my22itbziah5vaueoqdrfvzeofibu6wiijmzr7.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cm/ccmu2qqzhv2t6mvencq6x6rytfqnfjuwcf6chga5hzzrt4thshm6.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_10 = async_compile.triton('triton_poi_fused_cat_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s4/cs43l2m6kta2lj6onokdghygollhqhtzzl3xpdbuqkyzelmszlxp.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wo/cwouvxozhp7uhgdxokty22ohicfe5mrlgt6cnnoyacyjc6evc5ki.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zo/czox22un7msigswllza46qjp3nliszounvas2mprsxmndemcg2hj.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_13 = async_compile.triton('triton_poi_fused_cat_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vm/cvmqi2c7sowg4ztkbhmaqkodndn67hel2my7cc45z2lzltjb3f5n.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/55/c55slhhl3hdaqv77hknkmodhiniqwvdydfan77tapcthziq36xdy.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uk/cukfgspej2k35bqm6zylf2h7esaxjpbk5aepalcknpby5p4pj7co.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6k/c6ktzbel6zunt7acmrr4bkh3e5eeo6biv4rcvlqpppteix4lqcxd.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_poi_fused_cat_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sn/csn3yhylpiqoebhxh2pyw5jtdate6dqquxpfq54eqqpj7hkf4vpa.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_18 = async_compile.triton('triton_poi_fused_cat_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/az/caz4lbgput72xxykhs53oyyolb7sft5bw6ybdqtw3nfmzz67xeoz.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr74imknvo7wnlnohmdgmco7c5eotma67hh32re5aymdn47lr2v7.py
# Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv_ws_disp_d => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_70, %slice_74, %slice_78, %slice_82, %slice_86, %slice_90, %slice_94, %slice_98, %slice_102, %slice_106, %slice_110, %slice_114, %slice_118, %slice_122, %slice_126, %slice_130], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (16*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d3/cd3vbc4qy7e7qwikd7jtystlrvji7umj5zzbvi3qkrpi7oi6qtyv.py
# Topologically Sorted Source Nodes: [local_cv], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   local_cv => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %cat_5, %cat_8], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 48)
    x1 = xindex // 48
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 32, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (16*x1 + ((-16) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (16*x1 + ((-32) + x0)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (64, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (64, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf16 = empty_strided_cuda((4, 16, 4, 4), (256, 1, 64, 16), torch.float32)
        buf0 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 0)  # alias
        buf1 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 1)  # alias
        buf2 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 2)  # alias
        buf3 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 3)  # alias
        buf4 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 4)  # alias
        buf5 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 5)  # alias
        buf6 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 6)  # alias
        buf7 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 7)  # alias
        buf8 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 8)  # alias
        buf9 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 9)  # alias
        buf10 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 10)  # alias
        buf11 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 11)  # alias
        buf12 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 12)  # alias
        buf13 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 13)  # alias
        buf14 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 14)  # alias
        buf15 = reinterpret_tensor(buf16, (4, 1, 4, 4), (256, 1, 64, 16), 15)  # alias
        buf34 = empty_strided_cuda((4, 16, 4, 4), (256, 1, 64, 16), torch.float32)
        buf18 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 0)  # alias
        buf19 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 1)  # alias
        buf20 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 2)  # alias
        buf21 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 3)  # alias
        buf22 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 4)  # alias
        buf23 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 5)  # alias
        buf24 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 6)  # alias
        buf25 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 7)  # alias
        buf26 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 8)  # alias
        buf27 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 9)  # alias
        buf28 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 10)  # alias
        buf29 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 11)  # alias
        buf30 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 12)  # alias
        buf31 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 13)  # alias
        buf32 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 14)  # alias
        buf33 = reinterpret_tensor(buf34, (4, 1, 4, 4), (256, 1, 64, 16), 15)  # alias
        buf52 = empty_strided_cuda((4, 16, 4, 4), (256, 1, 64, 16), torch.float32)
        buf36 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 0)  # alias
        buf37 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 1)  # alias
        buf38 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 2)  # alias
        buf39 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 3)  # alias
        buf40 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 4)  # alias
        buf41 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 5)  # alias
        buf42 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 6)  # alias
        buf43 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 7)  # alias
        buf44 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 8)  # alias
        buf45 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 9)  # alias
        buf46 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 10)  # alias
        buf47 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 11)  # alias
        buf48 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 12)  # alias
        buf49 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 13)  # alias
        buf50 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 14)  # alias
        buf51 = reinterpret_tensor(buf52, (4, 1, 4, 4), (256, 1, 64, 16), 15)  # alias
        # Topologically Sorted Source Nodes: [add, tile_disp, mul_1, add_1, mul_2, add_2, mul_3, add_3, mul_4, add_4, mul_5, add_5, mul_6, add_6, mul_7, add_7, mul_8, add_8, mul_9, add_9, mul_10, add_10, mul_11, add_11, mul_12, add_12, mul_13, add_13, mul_14, add_14, mul_15, add_15, mul_16, add_16, mul_17, add_17, mul_18, add_18, mul_19, add_19, mul_20, add_20, mul_21, add_21, mul_22, add_22, mul_23, add_23, mul_24, add_24, mul_25, add_25, mul_26, add_26, mul_27, add_27, mul_28, add_28, mul_29, add_29, mul_30, add_30, mul_31, add_31, mul_32, add_32, add_33, tile_disp_1, mul_36, add_34, mul_37, add_35, mul_38, add_36, mul_39, add_37, mul_40, add_38, mul_41, add_39, mul_42, add_40, mul_43, add_41, mul_44, add_42, mul_45, add_43, mul_46, add_44, mul_47, add_45, mul_48, add_46, mul_49, add_47, mul_50, add_48, mul_51, add_49, mul_52, add_50, mul_53, add_51, mul_54, add_52, mul_55, add_53, mul_56, add_54, mul_57, add_55, mul_58, add_56, mul_59, add_57, mul_60, add_58, mul_61, add_59, mul_62, add_60, mul_63, add_61, mul_64, add_62, mul_65, add_63, mul_66, add_64, mul_67, add_65, add_66, tile_disp_2, mul_71, add_67, mul_72, add_68, mul_73, add_69, mul_74, add_70, mul_75, add_71, mul_76, add_72, mul_77, add_73, mul_78, add_74, mul_79, add_75, mul_80, add_76, mul_81, add_77, mul_82, add_78, mul_83, add_79, mul_84, add_80, mul_85, add_81, mul_86, add_82, mul_87, add_83, mul_88, add_84, mul_89, add_85, mul_90, add_86, mul_91, add_87, mul_92, add_88, mul_93, add_89, mul_94, add_90, mul_95, add_91, mul_96, add_92, mul_97, add_93, mul_98, add_94, mul_99, add_95, mul_100, add_96, mul_101, add_97, mul_102, add_98], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_0.run(arg0_1, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, 64, grid=grid(64), stream=stream0)
        del arg0_1
        buf17 = empty_strided_cuda((64, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1, vgrid, sub, setitem, clone, mul_33, truediv, sub_1, setitem_1], Original ATen: [aten.cat, aten._to_copy, aten.sub, aten.copy, aten.clone, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1.run(buf16, buf17, 2048, grid=grid(2048), stream=stream0)
        del buf0
        del buf1
        del buf10
        del buf11
        del buf12
        del buf13
        del buf14
        del buf15
        del buf2
        del buf3
        del buf4
        del buf5
        del buf6
        del buf7
        del buf8
        del buf9
        buf35 = empty_strided_cuda((64, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4, vgrid_2, sub_4, setitem_3, clone_2, mul_68, truediv_2, sub_5, setitem_4], Original ATen: [aten.cat, aten._to_copy, aten.sub, aten.copy, aten.clone, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1.run(buf34, buf35, 2048, grid=grid(2048), stream=stream0)
        del buf18
        del buf19
        del buf20
        del buf21
        del buf22
        del buf23
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        del buf30
        del buf31
        del buf32
        del buf33
        buf53 = empty_strided_cuda((64, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_7, vgrid_4, sub_8, setitem_6, clone_4, mul_103, truediv_4, sub_9, setitem_7], Original ATen: [aten.cat, aten._to_copy, aten.sub, aten.copy, aten.clone, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1.run(buf52, buf53, 2048, grid=grid(2048), stream=stream0)
        del buf36
        del buf37
        del buf38
        del buf39
        del buf40
        del buf41
        del buf42
        del buf43
        del buf44
        del buf45
        del buf46
        del buf47
        del buf48
        del buf49
        del buf50
        del buf51
        buf85 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf86 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf87 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf54 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf55 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf56 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf58 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf59 = buf58; del buf58  # reuse
        buf61 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf62 = buf61; del buf61  # reuse
        buf65 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf66 = buf65; del buf65  # reuse
        buf89 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf90 = buf89; del buf89  # reuse
        buf92 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf93 = buf92; del buf92  # reuse
        buf96 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [output, output_1], Original ATen: [aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_2.run(buf59, buf62, buf66, buf90, buf93, buf97, buf35, buf17, arg2_1, buf85, buf86, buf87, buf54, buf55, buf56, 4096, grid=grid(4096), stream=stream0)
        del buf17
        del buf35
        buf116 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf117 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.int64)
        buf118 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf120 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf121 = buf120; del buf120  # reuse
        buf123 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf124 = buf123; del buf123  # reuse
        buf127 = empty_strided_cuda((64, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.grid_sampler_2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_3.run(buf121, buf124, buf128, buf53, arg2_1, buf116, buf117, buf118, 4096, grid=grid(4096), stream=stream0)
        del buf53
        buf67 = reinterpret_tensor(buf52, (64, 4, 4), (16, 4, 1), 0); del buf52  # reuse
        buf98 = reinterpret_tensor(buf34, (64, 4, 4), (16, 4, 1), 0); del buf34  # reuse
        buf129 = reinterpret_tensor(buf16, (64, 4, 4), (16, 4, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [output, sub_3, norm, output_1, sub_7, norm_1, output_2, sub_11, norm_2], Original ATen: [aten.grid_sampler_2d, aten.sub, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_4.run(arg1_1, buf54, buf55, arg2_1, buf56, buf59, buf62, buf66, buf85, buf86, buf87, buf90, buf93, buf97, buf116, buf117, buf118, buf121, buf124, buf128, buf67, buf98, buf129, 1024, grid=grid(1024), stream=stream0)
        del arg1_1
        del arg2_1
        del buf116
        del buf117
        del buf118
        del buf121
        del buf124
        del buf128
        del buf54
        del buf55
        del buf56
        del buf59
        del buf62
        del buf66
        del buf85
        del buf86
        del buf87
        del buf90
        del buf93
        del buf97
        buf84 = empty_strided_cuda((64, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf68 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf67, buf68, 64, grid=grid(64), stream=stream0)
        buf69 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 1)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf67, buf69, 64, grid=grid(64), stream=stream0)
        buf70 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 2)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf67, buf70, 64, grid=grid(64), stream=stream0)
        buf71 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 3)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf67, buf71, 64, grid=grid(64), stream=stream0)
        buf72 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 4)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf67, buf72, 64, grid=grid(64), stream=stream0)
        buf73 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 5)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf67, buf73, 64, grid=grid(64), stream=stream0)
        buf74 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 6)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf67, buf74, 64, grid=grid(64), stream=stream0)
        buf75 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 7)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf67, buf75, 64, grid=grid(64), stream=stream0)
        buf76 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf67, buf76, 64, grid=grid(64), stream=stream0)
        buf77 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 9)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf67, buf77, 64, grid=grid(64), stream=stream0)
        buf78 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 10)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf67, buf78, 64, grid=grid(64), stream=stream0)
        buf79 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 11)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf67, buf79, 64, grid=grid(64), stream=stream0)
        buf80 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 12)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(buf67, buf80, 64, grid=grid(64), stream=stream0)
        buf81 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 13)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf67, buf81, 64, grid=grid(64), stream=stream0)
        buf82 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 14)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf67, buf82, 64, grid=grid(64), stream=stream0)
        buf83 = reinterpret_tensor(buf84, (64, 1, 1, 1), (16, 1, 1, 1), 15)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf67, buf83, 64, grid=grid(64), stream=stream0)
        buf115 = reinterpret_tensor(buf67, (64, 16, 1, 1), (16, 1, 1, 1), 0); del buf67  # reuse
        buf99 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf98, buf99, 64, grid=grid(64), stream=stream0)
        del buf68
        del buf69
        del buf70
        del buf71
        del buf72
        del buf73
        del buf74
        del buf75
        del buf76
        del buf77
        del buf78
        del buf79
        del buf80
        del buf81
        del buf82
        del buf83
        buf100 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 1)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf98, buf100, 64, grid=grid(64), stream=stream0)
        buf101 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 2)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf98, buf101, 64, grid=grid(64), stream=stream0)
        buf102 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 3)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf98, buf102, 64, grid=grid(64), stream=stream0)
        buf103 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 4)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf98, buf103, 64, grid=grid(64), stream=stream0)
        buf104 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 5)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf98, buf104, 64, grid=grid(64), stream=stream0)
        buf105 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 6)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf98, buf105, 64, grid=grid(64), stream=stream0)
        buf106 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 7)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf98, buf106, 64, grid=grid(64), stream=stream0)
        buf107 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf98, buf107, 64, grid=grid(64), stream=stream0)
        buf108 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 9)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf98, buf108, 64, grid=grid(64), stream=stream0)
        buf109 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 10)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf98, buf109, 64, grid=grid(64), stream=stream0)
        buf110 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 11)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf98, buf110, 64, grid=grid(64), stream=stream0)
        buf111 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 12)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(buf98, buf111, 64, grid=grid(64), stream=stream0)
        buf112 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 13)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf98, buf112, 64, grid=grid(64), stream=stream0)
        buf113 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 14)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf98, buf113, 64, grid=grid(64), stream=stream0)
        buf114 = reinterpret_tensor(buf115, (64, 1, 1, 1), (16, 1, 1, 1), 15)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf98, buf114, 64, grid=grid(64), stream=stream0)
        buf146 = reinterpret_tensor(buf98, (64, 16, 1, 1), (16, 1, 1, 1), 0); del buf98  # reuse
        buf130 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf129, buf130, 64, grid=grid(64), stream=stream0)
        del buf100
        del buf101
        del buf102
        del buf103
        del buf104
        del buf105
        del buf106
        del buf107
        del buf108
        del buf109
        del buf110
        del buf111
        del buf112
        del buf113
        del buf114
        del buf99
        buf131 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 1)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf129, buf131, 64, grid=grid(64), stream=stream0)
        buf132 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 2)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf129, buf132, 64, grid=grid(64), stream=stream0)
        buf133 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 3)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf129, buf133, 64, grid=grid(64), stream=stream0)
        buf134 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 4)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf129, buf134, 64, grid=grid(64), stream=stream0)
        buf135 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 5)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_10.run(buf129, buf135, 64, grid=grid(64), stream=stream0)
        buf136 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 6)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf129, buf136, 64, grid=grid(64), stream=stream0)
        buf137 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 7)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf129, buf137, 64, grid=grid(64), stream=stream0)
        buf138 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(buf129, buf138, 64, grid=grid(64), stream=stream0)
        buf139 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 9)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_14.run(buf129, buf139, 64, grid=grid(64), stream=stream0)
        buf140 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 10)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf129, buf140, 64, grid=grid(64), stream=stream0)
        buf141 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 11)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf129, buf141, 64, grid=grid(64), stream=stream0)
        buf142 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 12)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(buf129, buf142, 64, grid=grid(64), stream=stream0)
        buf143 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 13)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(buf129, buf143, 64, grid=grid(64), stream=stream0)
        buf144 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 14)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf129, buf144, 64, grid=grid(64), stream=stream0)
        buf145 = reinterpret_tensor(buf146, (64, 1, 1, 1), (16, 1, 1, 1), 15)  # alias
        # Topologically Sorted Source Nodes: [local_cv_ws_disp_d_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf129, buf145, 64, grid=grid(64), stream=stream0)
        del buf129
        buf147 = empty_strided_cuda((64, 48, 1, 1), (48, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [local_cv], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf84, buf115, buf146, buf147, 3072, grid=grid(3072), stream=stream0)
        del buf115
        del buf130
        del buf131
        del buf132
        del buf133
        del buf134
        del buf135
        del buf136
        del buf137
        del buf138
        del buf139
        del buf140
        del buf141
        del buf142
        del buf143
        del buf144
        del buf145
        del buf146
        del buf84
    return (buf147, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
