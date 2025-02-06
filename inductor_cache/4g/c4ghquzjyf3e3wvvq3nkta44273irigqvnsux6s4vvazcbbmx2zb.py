
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_threshold_backward_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 4)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    tmp28 = tl.load(in_ptr6 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp31 = tl.load(in_ptr7 + (0))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp41 = tl.load(in_ptr8 + (0))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr9 + (0))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp53 = tl.load(in_ptr11 + (0))
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK])
    tmp56 = tl.load(in_ptr12 + (0))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp66 = tl.load(in_ptr13 + (0))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp69 = tl.load(in_ptr14 + (0))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 32*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 3, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + (x0 + 16*x2), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp27 - tmp29
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp30 * tmp39
    tmp43 = tmp40 * tmp42
    tmp46 = tmp43 + tmp45
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp26, tmp46, tmp47)
    tmp49 = tmp0 >= tmp24
    tmp50 = tl.full([1], 4, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tl.load(in_ptr10 + (x0 + 16*x2), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp52 - tmp54
    tmp58 = 1e-05
    tmp59 = tmp57 + tmp58
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tl.full([1], 1, tl.int32)
    tmp62 = tmp61 / tmp60
    tmp63 = 1.0
    tmp64 = tmp62 * tmp63
    tmp65 = tmp55 * tmp64
    tmp68 = tmp65 * tmp67
    tmp71 = tmp68 + tmp70
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp49, tmp71, tmp72)
    tmp74 = tl.where(tmp26, tmp48, tmp73)
    tmp75 = tl.where(tmp4, tmp22, tmp74)
    tmp76 = tl.full([1], 0, tl.int32)
    tmp77 = triton_helpers.maximum(tmp76, tmp75)
    tmp78 = 0.0
    tmp79 = tmp77 <= tmp78
    tl.store(in_out_ptr0 + (x3), tmp77, xmask)
    tl.store(out_ptr0 + (x3), tmp79, xmask)
