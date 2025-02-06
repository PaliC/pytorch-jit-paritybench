
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_remainder_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy_div_mul_remainder_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 3)
    x0 = (xindex % 16)
    x2 = xindex // 48
    x3 = xindex
    tmp6 = tl.load(in_ptr0 + (x0 + 48*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (16 + x0 + 48*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (x3), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp5 = tmp3 == tmp3
    tmp8 = 255.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.002777777777777778
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.where(tmp5, tmp12, tmp6)
    tmp14 = 1.0
    tmp15 = tmp13 % tmp14
    tmp16 = tmp15 != tmp3
    tmp17 = (libdevice.signbit(tmp15) != 0) if (tmp15).dtype is tl.float32 else tmp15 < 0
    tmp18 = (libdevice.signbit(tmp14) != 0) if (tmp14).dtype is tl.float32 else tmp14 < 0
    tmp19 = tmp17 != tmp18
    tmp20 = tmp16 & tmp19
    tmp21 = tmp15 + tmp14
    tmp22 = tl.where(tmp20, tmp21, tmp15)
    tmp24 = tl.where(tmp4, tmp12, tmp23)
    tmp25 = tl.where(tmp4, tmp22, tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp0 == tmp3
    tmp30 = tl.where(tmp28, tmp12, tmp29)
    tmp31 = tl.where(tmp28, tmp22, tmp30)
    tmp32 = tl.where(tmp2, tmp27, tmp31)
    tl.store(out_ptr0 + (x3), tmp32, xmask)
