
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_adaptive_max_pool2d_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_adaptive_max_pool2d_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (11*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 11*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 11*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 11*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 11*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 11*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 11*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 11*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 11*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 11*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 11*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp21 = tmp1 > tmp0
    tmp22 = tl.full([1], 1, tl.int8)
    tmp23 = tl.full([1], 0, tl.int8)
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = tmp3 > tmp2
    tmp26 = tl.full([1], 2, tl.int8)
    tmp27 = tl.where(tmp25, tmp26, tmp24)
    tmp28 = tmp5 > tmp4
    tmp29 = tl.full([1], 3, tl.int8)
    tmp30 = tl.where(tmp28, tmp29, tmp27)
    tmp31 = tmp7 > tmp6
    tmp32 = tl.full([1], 4, tl.int8)
    tmp33 = tl.where(tmp31, tmp32, tmp30)
    tmp34 = tmp9 > tmp8
    tmp35 = tl.full([1], 5, tl.int8)
    tmp36 = tl.where(tmp34, tmp35, tmp33)
    tmp37 = tmp11 > tmp10
    tmp38 = tl.full([1], 6, tl.int8)
    tmp39 = tl.where(tmp37, tmp38, tmp36)
    tmp40 = tmp13 > tmp12
    tmp41 = tl.full([1], 7, tl.int8)
    tmp42 = tl.where(tmp40, tmp41, tmp39)
    tmp43 = tmp15 > tmp14
    tmp44 = tl.full([1], 8, tl.int8)
    tmp45 = tl.where(tmp43, tmp44, tmp42)
    tmp46 = tmp17 > tmp16
    tmp47 = tl.full([1], 9, tl.int8)
    tmp48 = tl.where(tmp46, tmp47, tmp45)
    tmp49 = tmp19 > tmp18
    tmp50 = tl.full([1], 10, tl.int8)
    tmp51 = tl.where(tmp49, tmp50, tmp48)
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp51, xmask)
