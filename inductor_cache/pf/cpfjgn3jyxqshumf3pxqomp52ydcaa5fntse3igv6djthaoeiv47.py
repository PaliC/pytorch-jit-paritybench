
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_reflection_pad2d_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_reflection_pad2d_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 591872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 34) % 34)
    x0 = (xindex % 34)
    x4 = xindex // 1156
    x2 = ((xindex // 1156) % 128)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (31 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x1)))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (31 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0)))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 16*tmp4 + 256*x4), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 256.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tl.load(in_ptr6 + (tmp8 + 16*tmp4 + 256*x4), xmask, eviction_policy='evict_last')
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x7), tmp24, xmask)
