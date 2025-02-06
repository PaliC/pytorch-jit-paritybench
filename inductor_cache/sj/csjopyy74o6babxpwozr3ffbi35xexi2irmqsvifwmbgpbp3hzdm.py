
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 2)
    x0 = (xindex % 16)
    x2 = xindex // 32
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (4*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr0 + (16 + x0 + 64*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr1 + (1 + 4*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 * tmp9
    tmp11 = triton_helpers.maximum(tmp7, tmp10)
    tmp12 = tl.load(in_ptr0 + (32 + x0 + 64*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr1 + (2 + 4*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = triton_helpers.maximum(tmp11, tmp14)
    tmp16 = tl.load(in_ptr0 + (48 + x0 + 64*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr1 + (3 + 4*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = triton_helpers.maximum(tmp15, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = tl.full([1], 2, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (x0 + 64*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr1 + (4*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr0 + (16 + x0 + 64*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr1 + (1 + 4*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp27 + tmp30
    tmp32 = tl.load(in_ptr0 + (32 + x0 + 64*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr1 + (2 + 4*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.load(in_ptr0 + (48 + x0 + 64*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr1 + (3 + 4*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp35 + tmp38
    tmp40 = 4.0
    tmp41 = tmp39 / tmp40
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp22, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp21, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, xmask)
