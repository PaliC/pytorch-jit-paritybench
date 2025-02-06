
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_max_unpool2d_10', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_unpool2d_10(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x4 = xindex // 2
    x1 = ((xindex // 2) % 2)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x4), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (2*((x3 % 2)) + 8*(x3 // 2)), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (1 + 2*((x3 % 2)) + 8*(x3 // 2)), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (4 + 2*((x3 % 2)) + 8*(x3 // 2)), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (5 + 2*((x3 % 2)) + 8*(x3 // 2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tl.where((tmp15 < 0) != (tmp17 < 0), tl.where(tmp15 % tmp17 != 0, tmp15 // tmp17 - 1, tmp15 // tmp17), tmp15 // tmp17)
    tmp19 = tmp18 * tmp17
    tmp20 = tmp15 - tmp19
    tmp21 = 2*x1
    tmp22 = tmp21 + tmp18
    tmp23 = 2*x0
    tmp24 = tmp23 + tmp20
    tmp25 = tl.full([1], 4, tl.int64)
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = 16*(x3 // 4)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full([XBLOCK], 8192, tl.int32)
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 8192)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 8192")
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tmp41 = triton_helpers.maximum(tmp40, tmp39)
    tl.store(out_ptr0 + (x3), tmp27, xmask)
    tl.store(out_ptr1 + (tl.broadcast_to(tmp33, [XBLOCK])), tmp41, xmask)
