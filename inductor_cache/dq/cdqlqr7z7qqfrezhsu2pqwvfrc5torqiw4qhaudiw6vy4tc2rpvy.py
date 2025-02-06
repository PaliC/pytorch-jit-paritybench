
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 10) % 10)
    x0 = (xindex % 10)
    x2 = ((xindex // 100) % 2)
    x3 = xindex // 200
    x6 = xindex
    tmp0 = (-3) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = x2
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + ((-15) + x0 + 4*x1 + 64*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr0 + (1 + x0 + 4*x1 + 64*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr0 + (17 + x0 + 4*x1 + 64*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.load(in_ptr0 + (33 + x0 + 4*x1 + 64*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = 4.0
    tmp25 = tmp23 / tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp16, tmp25, tmp26)
    tmp28 = tmp11 >= tmp14
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp11 < tmp29
    tmp31 = tmp28 & tmp10
    tmp32 = tl.load(in_ptr0 + ((-15) + x0 + 4*x1 + 64*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (1 + x0 + 4*x1 + 64*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = triton_helpers.maximum(tmp32, tmp33)
    tmp35 = tl.load(in_ptr0 + (17 + x0 + 4*x1 + 64*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = triton_helpers.maximum(tmp34, tmp35)
    tmp37 = tl.load(in_ptr0 + (33 + x0 + 4*x1 + 64*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = triton_helpers.maximum(tmp36, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp31, tmp38, tmp39)
    tmp41 = tl.where(tmp15, tmp27, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp10, tmp41, tmp42)
    tl.store(out_ptr0 + (x6), tmp43, xmask)
