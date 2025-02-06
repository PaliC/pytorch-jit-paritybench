
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_log_mul_rsub_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_log_mul_rsub_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 256)
    x0 = (xindex % 16)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (16 + x0 + 64*x2), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr2 + (32 + x0 + 64*x2), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr2 + (48 + x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = 1e-07
    tmp3 = tmp1 + tmp2
    tmp4 = tl_math.log(tmp3)
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 + tmp2
    tmp9 = tl_math.log(tmp8)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 - tmp10
    tmp12 = tmp3 - tmp8
    tmp13 = tmp12 + tmp2
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp12 * tmp14
    tmp16 = tmp11 - tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = 20.0
    tmp21 = tmp19 > tmp20
    tmp22 = tl_math.exp(tmp19)
    tmp23 = libdevice.log1p(tmp22)
    tmp24 = tmp23 * tmp18
    tmp25 = tl.where(tmp21, tmp17, tmp24)
    tmp26 = 0.0001
    tmp27 = tmp25 + tmp26
    tmp29 = tmp28 * tmp18
    tmp30 = tmp29 > tmp20
    tmp31 = tl_math.exp(tmp29)
    tmp32 = libdevice.log1p(tmp31)
    tmp33 = tmp32 * tmp18
    tmp34 = tl.where(tmp30, tmp28, tmp33)
    tmp35 = tmp34 + tmp26
    tmp36 = tmp27 + tmp35
    tmp37 = tmp27 / tmp36
    tmp38 = triton_helpers.maximum(tmp37, tmp26)
    tmp39 = triton_helpers.minimum(tmp38, tmp18)
    tmp40 = tl_math.log(tmp39)
    tmp41 = tmp7 * tmp40
    tmp42 = tmp16 + tmp41
    tmp43 = tl.full([1], 255, tl.int64)
    tmp44 = tmp43 - tmp6
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp18 - tmp37
    tmp47 = triton_helpers.maximum(tmp46, tmp26)
    tmp48 = triton_helpers.minimum(tmp47, tmp18)
    tmp49 = tl_math.log(tmp48)
    tmp50 = tmp45 * tmp49
    tmp51 = tmp42 + tmp50
    tmp53 = tmp52 * tmp18
    tmp54 = tmp53 > tmp20
    tmp55 = tl_math.exp(tmp53)
    tmp56 = libdevice.log1p(tmp55)
    tmp57 = tmp56 * tmp18
    tmp58 = tl.where(tmp54, tmp52, tmp57)
    tmp59 = tmp58 + tmp26
    tmp61 = tmp60 * tmp18
    tmp62 = tmp61 > tmp20
    tmp63 = tl_math.exp(tmp61)
    tmp64 = libdevice.log1p(tmp63)
    tmp65 = tmp64 * tmp18
    tmp66 = tl.where(tmp62, tmp60, tmp65)
    tmp67 = tmp66 + tmp26
    tmp68 = tmp59 + tmp67
    tmp69 = tmp59 / tmp68
    tmp70 = 49.9999999
    tmp71 = tmp69 * tmp70
    tmp72 = tmp71 + tmp2
    tmp73 = 0.0
    tmp74 = tmp72 >= tmp73
    tmp75 = -1.0
    tmp76 = tl.where(tmp74, tmp18, tmp75)
    tmp77 = tmp51 * tmp76
    tl.store(in_out_ptr0 + (x3), tmp77, None)
