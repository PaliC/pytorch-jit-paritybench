
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_sub_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_sub_0(in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12 * tmp4
    tmp14 = tmp13 - tmp2
    tmp15 = triton_helpers.maximum(tmp14, tmp7)
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1], 3, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tl.load(in_ptr0 + (tmp20 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp22 = tmp21 > tmp7
    tmp23 = 0.2
    tmp24 = tmp21 * tmp23
    tmp25 = tl.where(tmp22, tmp21, tmp24)
    tmp26 = tl.load(in_ptr0 + (tmp16 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp27 = tmp26 > tmp7
    tmp28 = tmp26 * tmp23
    tmp29 = tl.where(tmp27, tmp26, tmp28)
    tmp30 = tmp25 - tmp29
    tmp31 = tmp9 + tmp17
    tmp32 = triton_helpers.minimum(tmp31, tmp19)
    tmp33 = tl.load(in_ptr0 + (tmp20 + 4*tmp32 + 16*x2), xmask, eviction_policy='evict_last')
    tmp34 = tmp33 > tmp7
    tmp35 = tmp33 * tmp23
    tmp36 = tl.where(tmp34, tmp33, tmp35)
    tmp37 = tl.load(in_ptr0 + (tmp16 + 4*tmp32 + 16*x2), xmask, eviction_policy='evict_last')
    tmp38 = tmp37 > tmp7
    tmp39 = tmp37 * tmp23
    tmp40 = tl.where(tmp38, tmp37, tmp39)
    tmp41 = tmp36 - tmp40
    tmp42 = tmp16.to(tl.float32)
    tmp43 = tmp15 - tmp42
    tmp44 = triton_helpers.maximum(tmp43, tmp7)
    tmp45 = triton_helpers.minimum(tmp44, tmp4)
    tmp46 = tmp41 * tmp45
    tmp47 = tmp40 + tmp46
    tmp48 = tmp30 * tmp45
    tmp49 = tmp29 + tmp48
    tmp50 = tmp47 - tmp49
    tmp51 = tmp9.to(tl.float32)
    tmp52 = tmp8 - tmp51
    tmp53 = triton_helpers.maximum(tmp52, tmp7)
    tmp54 = triton_helpers.minimum(tmp53, tmp4)
    tmp55 = tmp50 * tmp54
    tmp56 = tmp49 + tmp55
    tl.store(in_out_ptr1 + (x4), tmp56, xmask)
