
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = ((x0) % 2)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (2*((((x0) // 2) % 64)) + 128*x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl_math.sin(tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tmp5 >= tmp8
    tmp16 = tl.full([1], 2, tl.int64)
    tmp17 = tmp5 < tmp16
    tmp18 = tmp15 & tmp4
    tmp19 = tl.load(in_ptr0 + (1 + 2*((((x0) // 2) % 64)) + 128*x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl_math.cos(tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp9, tmp14, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp4, tmp23, tmp24)
    tmp26 = tmp0 >= tmp3
    tmp27 = tl.full([1], 256, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = (((-128) + x0) % 2)
    tmp30 = tl.full([1], 0, tl.int64)
    tmp31 = tmp29 >= tmp30
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tmp29 < tmp32
    tmp34 = tmp33 & tmp26
    tmp35 = tl.load(in_ptr1 + (2*(((((-128) + x0) // 2) % 64)) + 128*x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tl_math.sin(tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp34, tmp36, tmp37)
    tmp39 = tmp29 >= tmp32
    tmp40 = tl.full([1], 2, tl.int64)
    tmp41 = tmp29 < tmp40
    tmp42 = tmp39 & tmp26
    tmp43 = tl.load(in_ptr1 + (1 + 2*(((((-128) + x0) // 2) % 64)) + 128*x1), tmp42, eviction_policy='evict_last', other=0.0)
    tmp44 = tl_math.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp42, tmp44, tmp45)
    tmp47 = tl.where(tmp33, tmp38, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp26, tmp47, tmp48)
    tmp50 = tl.where(tmp4, tmp25, tmp49)
    tl.store(out_ptr0 + (x2), tmp50, None)
