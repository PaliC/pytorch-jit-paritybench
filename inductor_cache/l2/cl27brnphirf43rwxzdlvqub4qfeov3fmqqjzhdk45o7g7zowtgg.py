
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x3 = xindex // 3
    x2 = xindex // 9
    x1 = ((xindex // 3) % 3)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x3), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + (3*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 * tmp11
    tmp13 = tl.load(in_ptr1 + (1 + 3*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp16 = tl.load(in_ptr1 + (2 + 3*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp20 = 1e-12
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tmp10 / tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp9, tmp22, tmp23)
    tmp25 = tmp0 >= tmp7
    tmp26 = tl.full([1], 3, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr2 + (x3), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr1 + (3*x2 + (((2 + x1) % 3))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr1 + (3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.load(in_ptr1 + (1 + 3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 + tmp33
    tmp35 = tl.load(in_ptr1 + (2 + 3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = libdevice.sqrt(tmp37)
    tmp39 = 1e-12
    tmp40 = triton_helpers.maximum(tmp38, tmp39)
    tmp41 = tmp29 / tmp40
    tmp42 = tmp28 * tmp41
    tmp43 = tl.load(in_ptr3 + (x3), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr1 + (3*x2 + (((1 + x1) % 3))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 / tmp40
    tmp46 = tmp43 * tmp45
    tmp47 = tmp42 - tmp46
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp25, tmp47, tmp48)
    tmp50 = tl.where(tmp9, tmp24, tmp49)
    tmp51 = tl.where(tmp4, tmp5, tmp50)
    tl.store(out_ptr0 + (x5), tmp51, xmask)
