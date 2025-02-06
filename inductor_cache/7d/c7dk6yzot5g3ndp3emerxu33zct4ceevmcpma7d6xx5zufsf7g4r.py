
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'out_ptr27': '*fp32', 'out_ptr28': '*fp32', 'out_ptr29': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32', 'out_ptr36': '*fp32', 'out_ptr37': '*fp32', 'out_ptr38': '*fp32', 'out_ptr39': '*fp32', 'out_ptr40': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_cos_mul_sin_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_cos_mul_sin_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + (3))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp21 = tl.load(in_ptr1 + (4))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (5))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp31 = tl.load(in_ptr1 + (6))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp36 = tl.load(in_ptr1 + (7))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp41 = tl.load(in_ptr1 + (8))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp46 = tl.load(in_ptr1 + (9))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp51 = tl.load(in_ptr1 + (10))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp56 = tl.load(in_ptr1 + (11))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp61 = tl.load(in_ptr1 + (12))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp66 = tl.load(in_ptr1 + (13))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp71 = tl.load(in_ptr1 + (14))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK])
    tmp76 = tl.load(in_ptr1 + (15))
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
    tmp81 = tl.load(in_ptr1 + (16))
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK])
    tmp86 = tl.load(in_ptr1 + (17))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp91 = tl.load(in_ptr1 + (18))
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK])
    tmp96 = tl.load(in_ptr1 + (19))
    tmp97 = tl.broadcast_to(tmp96, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp4 = tl_math.sin(tmp3)
    tmp5 = tl_math.cos(tmp3)
    tmp8 = tmp0 * tmp7
    tmp9 = tl_math.sin(tmp8)
    tmp10 = tl_math.cos(tmp8)
    tmp13 = tmp0 * tmp12
    tmp14 = tl_math.sin(tmp13)
    tmp15 = tl_math.cos(tmp13)
    tmp18 = tmp0 * tmp17
    tmp19 = tl_math.sin(tmp18)
    tmp20 = tl_math.cos(tmp18)
    tmp23 = tmp0 * tmp22
    tmp24 = tl_math.sin(tmp23)
    tmp25 = tl_math.cos(tmp23)
    tmp28 = tmp0 * tmp27
    tmp29 = tl_math.sin(tmp28)
    tmp30 = tl_math.cos(tmp28)
    tmp33 = tmp0 * tmp32
    tmp34 = tl_math.sin(tmp33)
    tmp35 = tl_math.cos(tmp33)
    tmp38 = tmp0 * tmp37
    tmp39 = tl_math.sin(tmp38)
    tmp40 = tl_math.cos(tmp38)
    tmp43 = tmp0 * tmp42
    tmp44 = tl_math.sin(tmp43)
    tmp45 = tl_math.cos(tmp43)
    tmp48 = tmp0 * tmp47
    tmp49 = tl_math.sin(tmp48)
    tmp50 = tl_math.cos(tmp48)
    tmp53 = tmp0 * tmp52
    tmp54 = tl_math.sin(tmp53)
    tmp55 = tl_math.cos(tmp53)
    tmp58 = tmp0 * tmp57
    tmp59 = tl_math.sin(tmp58)
    tmp60 = tl_math.cos(tmp58)
    tmp63 = tmp0 * tmp62
    tmp64 = tl_math.sin(tmp63)
    tmp65 = tl_math.cos(tmp63)
    tmp68 = tmp0 * tmp67
    tmp69 = tl_math.sin(tmp68)
    tmp70 = tl_math.cos(tmp68)
    tmp73 = tmp0 * tmp72
    tmp74 = tl_math.sin(tmp73)
    tmp75 = tl_math.cos(tmp73)
    tmp78 = tmp0 * tmp77
    tmp79 = tl_math.sin(tmp78)
    tmp80 = tl_math.cos(tmp78)
    tmp83 = tmp0 * tmp82
    tmp84 = tl_math.sin(tmp83)
    tmp85 = tl_math.cos(tmp83)
    tmp88 = tmp0 * tmp87
    tmp89 = tl_math.sin(tmp88)
    tmp90 = tl_math.cos(tmp88)
    tmp93 = tmp0 * tmp92
    tmp94 = tl_math.sin(tmp93)
    tmp95 = tl_math.cos(tmp93)
    tmp98 = tmp0 * tmp97
    tmp99 = tl_math.sin(tmp98)
    tmp100 = tl_math.cos(tmp98)
    tl.store(out_ptr0 + (x0 + 2624*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 2624*x1), tmp4, xmask)
    tl.store(out_ptr2 + (x0 + 2624*x1), tmp5, xmask)
    tl.store(out_ptr3 + (x0 + 2624*x1), tmp9, xmask)
    tl.store(out_ptr4 + (x0 + 2624*x1), tmp10, xmask)
    tl.store(out_ptr5 + (x0 + 2624*x1), tmp14, xmask)
    tl.store(out_ptr6 + (x0 + 2624*x1), tmp15, xmask)
    tl.store(out_ptr7 + (x0 + 2624*x1), tmp19, xmask)
    tl.store(out_ptr8 + (x0 + 2624*x1), tmp20, xmask)
    tl.store(out_ptr9 + (x0 + 2624*x1), tmp24, xmask)
    tl.store(out_ptr10 + (x0 + 2624*x1), tmp25, xmask)
    tl.store(out_ptr11 + (x0 + 2624*x1), tmp29, xmask)
    tl.store(out_ptr12 + (x0 + 2624*x1), tmp30, xmask)
    tl.store(out_ptr13 + (x0 + 2624*x1), tmp34, xmask)
    tl.store(out_ptr14 + (x0 + 2624*x1), tmp35, xmask)
    tl.store(out_ptr15 + (x0 + 2624*x1), tmp39, xmask)
    tl.store(out_ptr16 + (x0 + 2624*x1), tmp40, xmask)
    tl.store(out_ptr17 + (x0 + 2624*x1), tmp44, xmask)
    tl.store(out_ptr18 + (x0 + 2624*x1), tmp45, xmask)
    tl.store(out_ptr19 + (x0 + 2624*x1), tmp49, xmask)
    tl.store(out_ptr20 + (x0 + 2624*x1), tmp50, xmask)
    tl.store(out_ptr21 + (x0 + 2624*x1), tmp54, xmask)
    tl.store(out_ptr22 + (x0 + 2624*x1), tmp55, xmask)
    tl.store(out_ptr23 + (x0 + 2624*x1), tmp59, xmask)
    tl.store(out_ptr24 + (x0 + 2624*x1), tmp60, xmask)
    tl.store(out_ptr25 + (x0 + 2624*x1), tmp64, xmask)
    tl.store(out_ptr26 + (x0 + 2624*x1), tmp65, xmask)
    tl.store(out_ptr27 + (x0 + 2624*x1), tmp69, xmask)
    tl.store(out_ptr28 + (x0 + 2624*x1), tmp70, xmask)
    tl.store(out_ptr29 + (x0 + 2624*x1), tmp74, xmask)
    tl.store(out_ptr30 + (x0 + 2624*x1), tmp75, xmask)
    tl.store(out_ptr31 + (x0 + 2624*x1), tmp79, xmask)
    tl.store(out_ptr32 + (x0 + 2624*x1), tmp80, xmask)
    tl.store(out_ptr33 + (x0 + 2624*x1), tmp84, xmask)
    tl.store(out_ptr34 + (x0 + 2624*x1), tmp85, xmask)
    tl.store(out_ptr35 + (x0 + 2624*x1), tmp89, xmask)
    tl.store(out_ptr36 + (x0 + 2624*x1), tmp90, xmask)
    tl.store(out_ptr37 + (x0 + 2624*x1), tmp94, xmask)
    tl.store(out_ptr38 + (x0 + 2624*x1), tmp95, xmask)
    tl.store(out_ptr39 + (x0 + 2624*x1), tmp99, xmask)
    tl.store(out_ptr40 + (x0 + 2624*x1), tmp100, xmask)
