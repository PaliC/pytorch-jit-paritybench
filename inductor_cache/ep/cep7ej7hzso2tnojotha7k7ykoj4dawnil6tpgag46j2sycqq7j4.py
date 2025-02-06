
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1072812
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 299) % 299)
    x0 = (xindex % 299)
    x2 = xindex // 89401
    x3 = (xindex % 89401)
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.013377926421404682
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 + tmp2
    tmp17 = tmp16 * tmp4
    tmp18 = tmp17 - tmp2
    tmp19 = triton_helpers.maximum(tmp18, tmp7)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tmp20 + tmp10
    tmp22 = triton_helpers.minimum(tmp21, tmp12)
    tmp23 = tl.load(in_ptr0 + (tmp22 + 4*tmp13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (tmp20 + 4*tmp13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20.to(tl.float32)
    tmp27 = tmp19 - tmp26
    tmp28 = triton_helpers.maximum(tmp27, tmp7)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tl.load(in_ptr0 + (tmp20 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (tmp22 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp34 = tmp33 - tmp32
    tmp35 = tmp34 * tmp30
    tmp36 = tmp32 + tmp35
    tmp37 = tmp24 + tmp31
    tmp38 = tmp37 - tmp36
    tmp39 = tmp9.to(tl.float32)
    tmp40 = tmp8 - tmp39
    tmp41 = triton_helpers.maximum(tmp40, tmp7)
    tmp42 = triton_helpers.minimum(tmp41, tmp29)
    tmp43 = tmp38 * tmp42
    tl.store(out_ptr0 + (x3 + 89408*x2), tmp36, xmask)
    tl.store(in_out_ptr0 + (x3 + 89408*x2), tmp43, xmask)
