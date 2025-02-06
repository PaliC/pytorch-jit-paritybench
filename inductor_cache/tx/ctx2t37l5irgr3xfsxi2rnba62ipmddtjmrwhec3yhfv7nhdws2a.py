
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_out_ptr8': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_leaky_relu_native_group_norm_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 16, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_leaky_relu_native_group_norm_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex
    r2 = rindex // 16
    tmp0 = tl.load(in_out_ptr0 + (r3 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr2 + (r3 + 64*x0), xmask, other=0.0)
    tmp25 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_out_ptr4 + (r3 + 64*x0), xmask, other=0.0)
    tmp45 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_out_ptr6 + (r3 + 64*x0), xmask, other=0.0)
    tmp65 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr7 + (r2), None, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr8 + (r2), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr9 + (r2), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr10 + (r2), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr11 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(xmask, tmp27, 0)
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = tmp33 / tmp11
    tmp35 = tmp27 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None]
    tmp41 = tmp40 / tmp19
    tmp42 = tmp41 + tmp21
    tmp43 = libdevice.rsqrt(tmp42)
    tmp46 = tmp44 + tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp49 = tl.where(xmask, tmp47, 0)
    tmp50 = tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])
    tmp52 = tl.where(xmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp54 = tmp53 / tmp11
    tmp55 = tmp47 - tmp54
    tmp56 = tmp55 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.where(xmask, tmp57, 0)
    tmp60 = tl.sum(tmp59, 1)[:, None]
    tmp61 = tmp60 / tmp19
    tmp62 = tmp61 + tmp21
    tmp63 = libdevice.rsqrt(tmp62)
    tmp66 = tmp64 + tmp65
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
    tmp69 = tl.where(xmask, tmp67, 0)
    tmp70 = tl.broadcast_to(tmp67, [XBLOCK, RBLOCK])
    tmp72 = tl.where(xmask, tmp70, 0)
    tmp73 = tl.sum(tmp72, 1)[:, None]
    tmp74 = tmp73 / tmp11
    tmp75 = tmp67 - tmp74
    tmp76 = tmp75 * tmp75
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp79 = tl.where(xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tmp81 = tmp80 / tmp19
    tmp82 = tmp81 + tmp21
    tmp83 = libdevice.rsqrt(tmp82)
    tmp84 = tmp2 - tmp12
    tmp85 = tmp84 * tmp23
    tmp87 = tmp85 * tmp86
    tmp89 = tmp87 + tmp88
    tmp90 = tmp46 - tmp54
    tmp91 = tmp90 * tmp63
    tmp93 = tmp91 * tmp92
    tmp95 = tmp93 + tmp94
    tmp96 = tmp66 - tmp74
    tmp97 = tmp96 * tmp83
    tmp99 = tmp97 * tmp98
    tmp101 = tmp99 + tmp100
    tmp102 = tmp26 - tmp34
    tmp103 = tmp102 * tmp43
    tmp105 = tmp103 * tmp104
    tmp107 = tmp105 + tmp106
    tmp108 = 0.0
    tmp109 = tmp89 > tmp108
    tmp110 = 0.2
    tmp111 = tmp89 * tmp110
    tmp112 = tl.where(tmp109, tmp89, tmp111)
    tmp113 = tmp112 + tmp108
    tmp114 = tmp95 > tmp108
    tmp115 = tmp95 * tmp110
    tmp116 = tl.where(tmp114, tmp95, tmp115)
    tmp117 = tmp113 + tmp116
    tmp118 = tmp101 > tmp108
    tmp119 = tmp101 * tmp110
    tmp120 = tl.where(tmp118, tmp101, tmp119)
    tmp121 = tmp117 + tmp120
    tmp122 = tmp107 > tmp108
    tmp123 = tmp107 * tmp110
    tmp124 = tl.where(tmp122, tmp107, tmp123)
    tmp125 = tmp121 + tmp124
    tl.store(in_out_ptr0 + (r3 + 64*x0), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, xmask)
    tl.store(in_out_ptr2 + (r3 + 64*x0), tmp26, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr3 + (x0), tmp43, xmask)
    tl.store(in_out_ptr4 + (r3 + 64*x0), tmp46, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr5 + (x0), tmp63, xmask)
    tl.store(in_out_ptr6 + (r3 + 64*x0), tmp66, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr7 + (x0), tmp83, xmask)
    tl.store(in_out_ptr8 + (r3 + 64*x0), tmp125, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp34, xmask)
    tl.store(out_ptr2 + (x0), tmp54, xmask)
    tl.store(out_ptr3 + (x0), tmp74, xmask)
