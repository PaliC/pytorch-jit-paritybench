
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 9)
    x4 = xindex // 576
    x5 = ((xindex // 9) % 16)
    x7 = (xindex % 144)
    x8 = xindex // 144
    x9 = xindex
    x1 = ((xindex // 9) % 4)
    x2 = ((xindex // 36) % 4)
    tmp0 = tl.load(in_ptr0 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x7 + 144*x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x7 + 144*x4), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr6 + (x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr6 + (9 + x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr7 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr7 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr7 + (x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr7 + (9 + x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 + tmp5
    tmp13 = tmp6 * tmp12
    tmp15 = tl.full([XBLOCK], 36, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert(((0 <= tmp18) & (tmp18 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 36")
    tmp20 = tl.load(in_ptr3 + (tmp18 + 36*x8), xmask, eviction_policy='evict_last')
    tmp21 = tmp13 * tmp20
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 - tmp3
    tmp26 = tmp5 - tmp25
    tmp28 = tmp27.to(tl.int64)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 - tmp10
    tmp31 = tmp5 - tmp30
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33 + tmp15
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tl.device_assert(((0 <= tmp36) & (tmp36 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 36")
    tmp38 = tl.load(in_ptr3 + (tmp36 + 36*x8), xmask, eviction_policy='evict_last')
    tmp39 = tmp32 * tmp38
    tmp40 = tmp21 + tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 - tmp3
    tmp44 = tmp43 + tmp5
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 - tmp10
    tmp48 = tmp5 - tmp47
    tmp49 = tmp44 * tmp48
    tmp51 = tl.full([1], 6, tl.int64)
    tmp52 = tmp50 * tmp51
    tmp54 = tmp52 + tmp53
    tmp55 = tmp54 + tmp15
    tmp56 = tmp54 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp54)
    tl.device_assert(((0 <= tmp57) & (tmp57 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp57 < 36")
    tmp59 = tl.load(in_ptr3 + (tmp57 + 36*x8), xmask, eviction_policy='evict_last')
    tmp60 = tmp49 * tmp59
    tmp61 = tmp40 + tmp60
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp63 - tmp3
    tmp65 = tmp5 - tmp64
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tmp67 - tmp10
    tmp69 = tmp68 + tmp5
    tmp70 = tmp65 * tmp69
    tmp72 = tmp71 * tmp51
    tmp74 = tmp72 + tmp73
    tmp75 = tmp74 + tmp15
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tl.device_assert(((0 <= tmp77) & (tmp77 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp77 < 36")
    tmp79 = tl.load(in_ptr3 + (tmp77 + 36*x8), xmask, eviction_policy='evict_last')
    tmp80 = tmp70 * tmp79
    tmp81 = tmp61 + tmp80
    tl.store(in_out_ptr0 + (x9), tmp81, xmask)
