
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_elu_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_elu_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp9 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp25 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tmp8 = tmp7 + tmp3
    tmp10 = tmp9 > tmp1
    tmp11 = tmp9 * tmp3
    tmp12 = libdevice.expm1(tmp11)
    tmp13 = tmp12 * tmp3
    tmp14 = tl.where(tmp10, tmp11, tmp13)
    tmp15 = tmp14 + tmp3
    tmp16 = tmp8 + tmp15
    tmp18 = tmp17 > tmp1
    tmp19 = tmp17 * tmp3
    tmp20 = libdevice.expm1(tmp19)
    tmp21 = tmp20 * tmp3
    tmp22 = tl.where(tmp18, tmp19, tmp21)
    tmp23 = tmp22 + tmp3
    tmp24 = tmp16 + tmp23
    tmp26 = tmp25 > tmp1
    tmp27 = tmp25 * tmp3
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp3
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tmp31 = tmp30 + tmp3
    tmp32 = tmp24 + tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
