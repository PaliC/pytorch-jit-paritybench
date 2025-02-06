
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 4, 8, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_clamp_cos_div_mul_rsub_sin_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clamp_cos_div_mul_rsub_sin_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp26 = tl.load(in_ptr2 + (1))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp35 = tl.load(in_ptr3 + (1))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp42 = tl.load(in_ptr2 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp51 = tl.load(in_ptr3 + (2))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp58 = tl.load(in_ptr2 + (3))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp67 = tl.load(in_ptr3 + (3))
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK])
    tmp2 = 4.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 - tmp7
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = 1.0
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = 3.141592653589793
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.cos(tmp14)
    tmp16 = tmp11 - tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp21 = tmp20 * tmp0
    tmp22 = tl_math.sin(tmp21)
    tmp23 = tmp18 * tmp22
    tmp24 = tl_math.cos(tmp21)
    tmp25 = tmp18 * tmp24
    tmp28 = tmp5 - tmp27
    tmp29 = triton_helpers.maximum(tmp28, tmp9)
    tmp30 = triton_helpers.minimum(tmp29, tmp11)
    tmp31 = tmp30 * tmp13
    tmp32 = tl_math.cos(tmp31)
    tmp33 = tmp11 - tmp32
    tmp34 = tmp33 * tmp17
    tmp37 = tmp36 * tmp0
    tmp38 = tl_math.sin(tmp37)
    tmp39 = tmp34 * tmp38
    tmp40 = tl_math.cos(tmp37)
    tmp41 = tmp34 * tmp40
    tmp44 = tmp5 - tmp43
    tmp45 = triton_helpers.maximum(tmp44, tmp9)
    tmp46 = triton_helpers.minimum(tmp45, tmp11)
    tmp47 = tmp46 * tmp13
    tmp48 = tl_math.cos(tmp47)
    tmp49 = tmp11 - tmp48
    tmp50 = tmp49 * tmp17
    tmp53 = tmp52 * tmp0
    tmp54 = tl_math.sin(tmp53)
    tmp55 = tmp50 * tmp54
    tmp56 = tl_math.cos(tmp53)
    tmp57 = tmp50 * tmp56
    tmp60 = tmp5 - tmp59
    tmp61 = triton_helpers.maximum(tmp60, tmp9)
    tmp62 = triton_helpers.minimum(tmp61, tmp11)
    tmp63 = tmp62 * tmp13
    tmp64 = tl_math.cos(tmp63)
    tmp65 = tmp11 - tmp64
    tmp66 = tmp65 * tmp17
    tmp69 = tmp68 * tmp0
    tmp70 = tl_math.sin(tmp69)
    tmp71 = tmp66 * tmp70
    tmp72 = tl_math.cos(tmp69)
    tmp73 = tmp66 * tmp72
    tl.store(out_ptr0 + (x0 + 36*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 36*x1), tmp23, xmask)
    tl.store(out_ptr2 + (x0 + 36*x1), tmp25, xmask)
    tl.store(out_ptr3 + (x0 + 36*x1), tmp39, xmask)
    tl.store(out_ptr4 + (x0 + 36*x1), tmp41, xmask)
    tl.store(out_ptr5 + (x0 + 36*x1), tmp55, xmask)
    tl.store(out_ptr6 + (x0 + 36*x1), tmp57, xmask)
    tl.store(out_ptr7 + (x0 + 36*x1), tmp71, xmask)
    tl.store(out_ptr8 + (x0 + 36*x1), tmp73, xmask)
