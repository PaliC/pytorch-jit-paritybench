
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x4 = xindex // 256
    x2 = ((xindex // 256) % 128)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 8*((x0 % 2)) + (x1)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (8*(8*((x0 % 2)) + (x1)) + 64*x4 + (x0 // 2)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tmp5 >= tmp8
    tmp17 = tl.full([1], 16, tl.int64)
    tmp18 = tmp5 < tmp17
    tmp19 = tmp16 & tmp4
    tmp20 = tl.load(in_ptr2 + (8*((-8) + 8*((x0 % 2)) + (x1)) + 64*x4 + (x0 // 2)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr3 + (x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp9, tmp15, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 16, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = 8*((x0 % 2)) + ((-8) + x1)
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 8, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp28
    tmp37 = tl.load(in_ptr4 + (8*(8*((x0 % 2)) + ((-8) + x1)) + 64*x4 + (x0 // 2)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp36, tmp39, tmp40)
    tmp42 = tmp31 >= tmp34
    tmp43 = tl.full([1], 16, tl.int64)
    tmp44 = tmp31 < tmp43
    tmp45 = tmp42 & tmp28
    tmp46 = tl.load(in_ptr6 + (8*((-8) + 8*((x0 % 2)) + ((-8) + x1)) + 64*x4 + (x0 // 2)), tmp45, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr7 + (x2), tmp45, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.where(tmp35, tmp41, tmp50)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp28, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp27, tmp53)
    tl.store(out_ptr0 + (x5), tmp54, None)
