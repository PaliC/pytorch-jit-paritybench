
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_linalg_vector_norm_mul_relu_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_div_linalg_vector_norm_mul_relu_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp20 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp29 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp1 = 0.0025
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tmp4 * tmp0
    tmp6 = tl_math.abs(tmp2)
    tmp7 = 1e-12
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 / tmp8
    tmp10 = tl_math.abs(tmp9)
    tmp12 = tmp11 - tmp1
    tmp13 = triton_helpers.maximum(tmp3, tmp12)
    tmp14 = tmp13 * tmp11
    tmp15 = tl_math.abs(tmp12)
    tmp16 = tmp15 + tmp7
    tmp17 = tmp14 / tmp16
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tmp10 + tmp18
    tmp21 = tmp20 - tmp1
    tmp22 = triton_helpers.maximum(tmp3, tmp21)
    tmp23 = tmp22 * tmp20
    tmp24 = tl_math.abs(tmp21)
    tmp25 = tmp24 + tmp7
    tmp26 = tmp23 / tmp25
    tmp27 = tl_math.abs(tmp26)
    tmp28 = tmp19 + tmp27
    tmp30 = tmp29 - tmp1
    tmp31 = triton_helpers.maximum(tmp3, tmp30)
    tmp32 = tmp31 * tmp29
    tmp33 = tl_math.abs(tmp30)
    tmp34 = tmp33 + tmp7
    tmp35 = tmp32 / tmp34
    tmp36 = tl_math.abs(tmp35)
    tmp37 = tmp28 + tmp36
    tl.store(out_ptr0 + (x2), tmp37, xmask)
