
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_ge_gt_le_lt_mul_pow_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_ge_gt_le_lt_mul_pow_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 * tmp0
    tmp5 = 0.25
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 - tmp6
    tmp8 = 0.0
    tmp9 = tmp0 >= tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp7 * tmp10
    tmp12 = tmp0 <= tmp1
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp3 + tmp14
    tmp16 = tmp0 + tmp6
    tmp17 = tmp0 < tmp8
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp20 = -2.0
    tmp21 = tmp0 >= tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp19 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp0 < tmp20
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 - tmp26
    tl.store(out_ptr0 + (x0), tmp27, xmask)
