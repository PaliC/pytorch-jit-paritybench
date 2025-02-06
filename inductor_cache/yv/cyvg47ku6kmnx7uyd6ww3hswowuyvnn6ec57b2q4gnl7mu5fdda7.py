
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_clamp_mul_pow_reciprocal_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_clamp_mul_pow_reciprocal_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1e-06
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = libdevice.pow(tmp2, tmp4)
    tmp7 = triton_helpers.maximum(tmp6, tmp1)
    tmp8 = libdevice.pow(tmp7, tmp4)
    tmp9 = tmp8 + tmp5
    tmp11 = triton_helpers.maximum(tmp10, tmp1)
    tmp12 = libdevice.pow(tmp11, tmp4)
    tmp13 = tmp12 + tmp9
    tmp15 = triton_helpers.maximum(tmp14, tmp1)
    tmp16 = libdevice.pow(tmp15, tmp4)
    tmp17 = tmp16 + tmp13
    tmp19 = triton_helpers.maximum(tmp18, tmp1)
    tmp20 = libdevice.pow(tmp19, tmp4)
    tmp21 = tmp20 + tmp17
    tmp23 = triton_helpers.maximum(tmp22, tmp1)
    tmp24 = libdevice.pow(tmp23, tmp4)
    tmp25 = tmp24 + tmp21
    tmp27 = triton_helpers.maximum(tmp26, tmp1)
    tmp28 = libdevice.pow(tmp27, tmp4)
    tmp29 = tmp28 + tmp25
    tmp31 = triton_helpers.maximum(tmp30, tmp1)
    tmp32 = libdevice.pow(tmp31, tmp4)
    tmp33 = tmp32 + tmp29
    tmp35 = triton_helpers.maximum(tmp34, tmp1)
    tmp36 = libdevice.pow(tmp35, tmp4)
    tmp37 = tmp36 + tmp33
    tmp39 = triton_helpers.maximum(tmp38, tmp1)
    tmp40 = libdevice.pow(tmp39, tmp4)
    tmp41 = tmp40 + tmp37
    tmp43 = triton_helpers.maximum(tmp42, tmp1)
    tmp44 = libdevice.pow(tmp43, tmp4)
    tmp45 = tmp44 + tmp41
    tmp47 = triton_helpers.maximum(tmp46, tmp1)
    tmp48 = libdevice.pow(tmp47, tmp4)
    tmp49 = tmp48 + tmp45
    tmp51 = triton_helpers.maximum(tmp50, tmp1)
    tmp52 = libdevice.pow(tmp51, tmp4)
    tmp53 = tmp52 + tmp49
    tmp55 = triton_helpers.maximum(tmp54, tmp1)
    tmp56 = libdevice.pow(tmp55, tmp4)
    tmp57 = tmp56 + tmp53
    tmp59 = triton_helpers.maximum(tmp58, tmp1)
    tmp60 = libdevice.pow(tmp59, tmp4)
    tmp61 = tmp60 + tmp57
    tmp63 = triton_helpers.maximum(tmp62, tmp1)
    tmp64 = libdevice.pow(tmp63, tmp4)
    tmp65 = tmp64 + tmp61
    tmp66 = 0.0625
    tmp67 = tmp65 * tmp66
    tmp68 = tl.full([1], 1, tl.int32)
    tmp69 = tmp68 / tmp4
    tmp70 = 1.0
    tmp71 = tmp69 * tmp70
    tmp72 = libdevice.pow(tmp67, tmp71)
    tl.store(in_out_ptr0 + (x0), tmp72, xmask)
