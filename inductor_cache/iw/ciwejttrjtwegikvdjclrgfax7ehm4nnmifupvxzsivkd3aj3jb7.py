
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i1', 'out_ptr3': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_floor_ge_gt_le_logical_and_lt_maximum_mul_neg_sign_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_floor_ge_gt_le_logical_and_lt_maximum_mul_neg_sign_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp28 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = tmp5 < tmp4
    tmp7 = tmp6.to(tl.int8)
    tmp8 = tmp4 < tmp5
    tmp9 = tmp8.to(tl.int8)
    tmp10 = tmp7 - tmp9
    tmp11 = tmp10.to(tmp4.dtype)
    tmp12 = tl_math.abs(tmp4)
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = tmp11 * tmp15
    tmp19 = triton_helpers.maximum(tmp16, tmp18)
    tmp22 = triton_helpers.minimum(tmp19, tmp21)
    tmp23 = tmp22 + tmp3
    tmp24 = tmp23 * tmp1
    tmp25 = tmp16 >= tmp18
    tmp26 = tmp16 <= tmp21
    tmp27 = tmp25 & tmp26
    tmp29 = tmp28 / tmp1
    tmp30 = tmp29 - tmp3
    tmp31 = tl_math.abs(tmp30)
    tmp33 = tmp32 / tmp1
    tmp34 = tmp33 - tmp3
    tmp35 = tl_math.abs(tmp34)
    tmp36 = triton_helpers.maximum(tmp31, tmp35)
    tmp37 = tmp4 > tmp36
    tmp38 = -tmp36
    tmp39 = tmp4 < tmp38
    tl.store(out_ptr0 + (x2), tmp24, xmask)
    tl.store(out_ptr1 + (x2), tmp27, xmask)
    tl.store(out_ptr2 + (x2), tmp37, xmask)
    tl.store(out_ptr3 + (x2), tmp39, xmask)
