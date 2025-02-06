
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_cosh_log_mul_pow_reciprocal_sum_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_cosh_log_mul_pow_reciprocal_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = (xindex % 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x4 + 64*x2), xmask)
    tmp3 = tl.load(in_ptr1 + (16 + x4 + 64*x2), xmask)
    tmp6 = tl.load(in_ptr1 + (32 + x4 + 64*x2), xmask)
    tmp9 = tl.load(in_ptr1 + (48 + x4 + 64*x2), xmask)
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr3 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp2 = tmp0 * tmp1
    tmp4 = tmp0 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp0 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp0 * tmp9
    tmp11 = tmp8 + tmp10
    tmp14 = tmp11 + tmp13
    tmp17 = libdevice.cosh(tmp14)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.full([1], 1, tl.int32)
    tmp20 = tmp19 / tmp18
    tmp21 = 1.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp23 + tmp21
    tmp25 = tl_math.abs(tmp24)
    tmp26 = tl_math.log(tmp25)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp26, xmask)
