
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_div_mean_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 12, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_div_mean_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp7 = tl.load(in_ptr0 + (16 + r0), None)
    tmp8 = tl.load(in_ptr1 + (16 + r0), None)
    tmp14 = tl.load(in_ptr0 + (32 + r0), None)
    tmp15 = tl.load(in_ptr1 + (32 + r0), None)
    tmp21 = tl.load(in_ptr0 + (64 + r0), None)
    tmp22 = tl.load(in_ptr1 + (64 + r0), None)
    tmp28 = tl.load(in_ptr0 + (80 + r0), None)
    tmp29 = tl.load(in_ptr1 + (80 + r0), None)
    tmp35 = tl.load(in_ptr0 + (96 + r0), None)
    tmp36 = tl.load(in_ptr1 + (96 + r0), None)
    tmp42 = tl.load(in_ptr0 + (128 + r0), None)
    tmp43 = tl.load(in_ptr1 + (128 + r0), None)
    tmp49 = tl.load(in_ptr0 + (144 + r0), None)
    tmp50 = tl.load(in_ptr1 + (144 + r0), None)
    tmp56 = tl.load(in_ptr0 + (160 + r0), None)
    tmp57 = tl.load(in_ptr1 + (160 + r0), None)
    tmp63 = tl.load(in_ptr0 + (192 + r0), None)
    tmp64 = tl.load(in_ptr1 + (192 + r0), None)
    tmp70 = tl.load(in_ptr0 + (208 + r0), None)
    tmp71 = tl.load(in_ptr1 + (208 + r0), None)
    tmp77 = tl.load(in_ptr0 + (224 + r0), None)
    tmp78 = tl.load(in_ptr1 + (224 + r0), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp9 = tmp7 - tmp8
    tmp10 = tl_math.abs(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tmp16 = tmp14 - tmp15
    tmp17 = tl_math.abs(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tmp23 = tmp21 - tmp22
    tmp24 = tl_math.abs(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp30 = tmp28 - tmp29
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.sum(tmp32, 1)[:, None]
    tmp37 = tmp35 - tmp36
    tmp38 = tl_math.abs(tmp37)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.sum(tmp39, 1)[:, None]
    tmp44 = tmp42 - tmp43
    tmp45 = tl_math.abs(tmp44)
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.sum(tmp46, 1)[:, None]
    tmp51 = tmp49 - tmp50
    tmp52 = tl_math.abs(tmp51)
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
    tmp55 = tl.sum(tmp53, 1)[:, None]
    tmp58 = tmp56 - tmp57
    tmp59 = tl_math.abs(tmp58)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, RBLOCK])
    tmp62 = tl.sum(tmp60, 1)[:, None]
    tmp65 = tmp63 - tmp64
    tmp66 = tl_math.abs(tmp65)
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
    tmp69 = tl.sum(tmp67, 1)[:, None]
    tmp72 = tmp70 - tmp71
    tmp73 = tl_math.abs(tmp72)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
    tmp76 = tl.sum(tmp74, 1)[:, None]
    tmp79 = tmp77 - tmp78
    tmp80 = tl_math.abs(tmp79)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp83 = tl.sum(tmp81, 1)[:, None]
    tmp84 = 16.0
    tmp85 = tmp6 / tmp84
    tmp86 = 0.0
    tmp87 = tmp85 + tmp86
    tmp88 = tmp13 / tmp84
    tmp89 = tmp87 + tmp88
    tmp90 = tmp20 / tmp84
    tmp91 = tmp89 + tmp90
    tmp92 = 0.3333333333333333
    tmp93 = tmp91 * tmp92
    tmp94 = tmp93 + tmp86
    tmp95 = tmp27 / tmp84
    tmp96 = tmp95 + tmp86
    tmp97 = tmp34 / tmp84
    tmp98 = tmp96 + tmp97
    tmp99 = tmp41 / tmp84
    tmp100 = tmp98 + tmp99
    tmp101 = tmp100 * tmp92
    tmp102 = tmp94 + tmp101
    tmp103 = tmp48 / tmp84
    tmp104 = tmp103 + tmp86
    tmp105 = tmp55 / tmp84
    tmp106 = tmp104 + tmp105
    tmp107 = tmp62 / tmp84
    tmp108 = tmp106 + tmp107
    tmp109 = tmp108 * tmp92
    tmp110 = tmp102 + tmp109
    tmp111 = tmp69 / tmp84
    tmp112 = tmp111 + tmp86
    tmp113 = tmp76 / tmp84
    tmp114 = tmp112 + tmp113
    tmp115 = tmp83 / tmp84
    tmp116 = tmp114 + tmp115
    tmp117 = tmp116 * tmp92
    tmp118 = tmp110 + tmp117
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp118, None)
