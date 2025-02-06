
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x2 = xindex // 12
    x3 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tl.load(in_ptr1 + (1 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.where(tmp3, tmp4, tmp8)
    tmp10 = tl.load(in_ptr0 + (2*x3), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + (1 + 2*x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.where(tmp3, tmp10, tmp11)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp2 >= tmp0
    tmp15 = tmp2 < tmp2
    tmp16 = tl.load(in_ptr0 + (2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp2 >= tmp2
    tmp18 = tmp2 < tmp6
    tmp19 = tl.load(in_ptr1 + (1 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp15, tmp16, tmp19)
    tmp21 = tl.load(in_ptr0 + (2*x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr1 + (1 + 2*x3), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp15, tmp21, tmp22)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp13 + tmp24
    tmp26 = tmp20 * tmp12
    tmp27 = tmp9 * tmp23
    tmp28 = tmp26 - tmp27
    tmp29 = tl.load(in_ptr0 + (6 + 2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr1 + (7 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp3, tmp29, tmp30)
    tmp32 = tmp31 * tmp12
    tmp33 = tl.load(in_ptr0 + (6 + 2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr1 + (7 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tmp35 * tmp23
    tmp37 = tmp32 + tmp36
    tmp38 = tmp35 * tmp12
    tmp39 = tmp31 * tmp23
    tmp40 = tmp38 - tmp39
    tmp41 = tl.load(in_ptr0 + (12 + 2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr1 + (13 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.where(tmp3, tmp41, tmp42)
    tmp44 = tmp43 * tmp12
    tmp45 = tl.load(in_ptr0 + (12 + 2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr1 + (13 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.where(tmp15, tmp45, tmp46)
    tmp48 = tmp47 * tmp23
    tmp49 = tmp44 + tmp48
    tmp50 = tmp47 * tmp12
    tmp51 = tmp43 * tmp23
    tmp52 = tmp50 - tmp51
    tmp53 = tl.load(in_ptr0 + (18 + 2*x0 + 24*x2), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr1 + (19 + 2*x0 + 24*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tl.where(tmp3, tmp53, tmp54)
    tmp56 = tmp55 * tmp12
    tmp57 = tl.load(in_ptr0 + (18 + 2*x0 + 24*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr1 + (19 + 2*x0 + 24*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tl.where(tmp15, tmp57, tmp58)
    tmp60 = tmp59 * tmp23
    tmp61 = tmp56 + tmp60
    tmp62 = tmp59 * tmp12
    tmp63 = tmp55 * tmp23
    tmp64 = tmp62 - tmp63
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
    tl.store(out_ptr2 + (x3), tmp37, xmask)
    tl.store(out_ptr3 + (x3), tmp40, xmask)
    tl.store(out_ptr4 + (x3), tmp49, xmask)
    tl.store(out_ptr5 + (x3), tmp52, xmask)
    tl.store(out_ptr6 + (x3), tmp61, xmask)
    tl.store(out_ptr7 + (x3), tmp64, xmask)
