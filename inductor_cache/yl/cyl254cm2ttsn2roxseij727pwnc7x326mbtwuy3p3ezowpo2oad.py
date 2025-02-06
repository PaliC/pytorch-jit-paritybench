
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_hardtanh_mean_mul_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_hardtanh_mean_mul_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 480)
    x1 = xindex // 480
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1920*x1), xmask)
    tmp10 = tl.load(in_ptr0 + (480 + x0 + 1920*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (960 + x0 + 1920*x1), xmask)
    tmp24 = tl.load(in_ptr0 + (1440 + x0 + 1920*x1), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp11, tmp3)
    tmp13 = triton_helpers.minimum(tmp12, tmp5)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 * tmp8
    tmp16 = tmp9 + tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = triton_helpers.minimum(tmp19, tmp5)
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp16 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = triton_helpers.maximum(tmp25, tmp3)
    tmp27 = triton_helpers.minimum(tmp26, tmp5)
    tmp28 = tmp24 * tmp27
    tmp29 = tmp28 * tmp8
    tmp30 = tmp23 + tmp29
    tmp31 = 4.0
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
