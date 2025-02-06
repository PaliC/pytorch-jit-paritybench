
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp16 = tl.load(in_ptr3 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp26 = tl.load(in_ptr5 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp45 = tl.load(in_ptr10 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp5 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x4 + 16*x3), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tmp0 >= tmp12
    tmp22 = tl.full([1], 3, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x4 + 16*x3), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp25 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp24, tmp28, tmp29)
    tmp31 = tmp0 >= tmp22
    tmp32 = tl.full([1], 4, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x4 + 16*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr7 + (x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full([XBLOCK], 1, tl.int32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tmp40 = tl.load(in_ptr8 + (x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 + tmp36
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp44 = tl.load(in_ptr9 + (x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp44 + tmp46
    tmp48 = tl.full([1], 0, tl.int32)
    tmp49 = triton_helpers.maximum(tmp48, tmp47)
    tmp50 = tl.load(in_ptr11 + (x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp36
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tmp49 - tmp49
    tmp55 = tl.load(in_ptr12 + (x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp49 + tmp56
    tmp58 = tmp57 - tmp34
    tmp59 = tl.load(in_ptr13 + (x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp58 * tmp59
    tmp61 = tmp34 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp31, tmp61, tmp62)
    tmp64 = tl.where(tmp24, tmp30, tmp63)
    tmp65 = tl.where(tmp14, tmp20, tmp64)
    tmp66 = tl.where(tmp4, tmp10, tmp65)
    tl.store(out_ptr0 + (x5), tmp66, xmask)
