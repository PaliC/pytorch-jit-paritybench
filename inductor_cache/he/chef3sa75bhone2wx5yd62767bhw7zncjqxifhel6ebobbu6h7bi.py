
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (12 + 8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (4 + 8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = -tmp6
    tmp8 = tmp5 + tmp7
    tmp9 = tl.load(in_ptr0 + (8 + 8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (8*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = -tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = libdevice.atan2(tmp8, tmp12)
    tmp14 = 1.5707963
    tmp15 = libdevice.fmod(tmp13, tmp14)
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tl_math.cos(tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 8, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr0 + (12 + 8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr0 + (4 + 8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = -tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = tl.load(in_ptr0 + (8 + 8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr0 + (8*x1 + ((-4) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = -tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = libdevice.atan2(tmp26, tmp30)
    tmp32 = 1.5707963
    tmp33 = libdevice.fmod(tmp31, tmp32)
    tmp34 = tl_math.abs(tmp33)
    tmp35 = tmp34 - tmp32
    tmp36 = tl_math.cos(tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp20, tmp36, tmp37)
    tmp39 = tl.where(tmp4, tmp19, tmp38)
    tmp40 = tmp16 + tmp14
    tmp41 = tl_math.cos(tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp4, tmp41, tmp42)
    tmp44 = tl_math.cos(tmp34)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp20, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp43, tmp46)
    tl.store(out_ptr0 + (x2), tmp39, xmask)
    tl.store(out_ptr1 + (x2), tmp47, xmask)
