
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_linalg_vector_norm_mul_reciprocal_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_linalg_vector_norm_mul_reciprocal_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x4 = (xindex % 16)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x4 + 64*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (16 + x4 + 64*x2), xmask)
    tmp8 = tl.load(in_ptr0 + (32 + x4 + 64*x2), xmask)
    tmp12 = tl.load(in_ptr0 + (48 + x4 + 64*x2), xmask)
    tmp21 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp5 = tmp4 - tmp1
    tmp6 = tl_math.abs(tmp5)
    tmp7 = tmp3 + tmp6
    tmp9 = tmp8 - tmp1
    tmp10 = tl_math.abs(tmp9)
    tmp11 = tmp7 + tmp10
    tmp13 = tmp12 - tmp1
    tmp14 = tl_math.abs(tmp13)
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = tmp19 * tmp16
    tmp22 = tl_math.abs(tmp21)
    tmp23 = tmp22 - tmp15
    tmp24 = tl_math.abs(tmp23)
    tmp25 = tmp24 + tmp16
    tmp26 = tmp18 / tmp25
    tmp27 = tmp26 * tmp16
    tmp28 = tmp20 - tmp27
    tl.store(in_out_ptr0 + (x5), tmp28, xmask)
