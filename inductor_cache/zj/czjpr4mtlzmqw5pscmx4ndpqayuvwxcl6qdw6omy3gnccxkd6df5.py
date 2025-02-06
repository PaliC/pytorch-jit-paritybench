
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 5)
    x0 = (xindex % 16)
    x2 = xindex // 80
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr0 + (x0 + 16*x2), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (x0 + 16*x2), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 - tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp14, tmp17, tmp18)
    tmp20 = tmp0 >= tmp12
    tmp21 = tl.full([1], 4, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr0 + (x0 + 16*x2), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr1 + (x0 + 16*x2), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = 0.1
    tmp27 = tmp25 + tmp26
    tmp28 = tmp24 / tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp23, tmp28, tmp29)
    tmp31 = tmp0 >= tmp21
    tmp32 = tl.full([1], 5, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr1 + (x0 + 16*x2), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr0 + (x0 + 16*x2), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = 0.1
    tmp37 = tmp35 + tmp36
    tmp38 = tmp34 / tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp31, tmp38, tmp39)
    tmp41 = tl.where(tmp23, tmp30, tmp40)
    tmp42 = tl.where(tmp14, tmp19, tmp41)
    tmp43 = tl.where(tmp9, tmp10, tmp42)
    tmp44 = tl.where(tmp4, tmp5, tmp43)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
