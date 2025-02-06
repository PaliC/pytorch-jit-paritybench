
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 32)
    x0 = (xindex % 64)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 512*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 256*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 16, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr0 + (x0 + 64*((-8) + x1) + 512*x2), tmp15, other=0.0)
    tmp17 = tl.load(in_ptr2 + (x0 + 64*((-8) + x1) + 512*x2), tmp15, other=0.0)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.load(in_ptr1 + (64 + x0 + 256*x2), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp18 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tmp0 >= tmp13
    tmp26 = tl.full([1], 24, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr0 + (x0 + 64*((-16) + x1) + 512*x2), tmp28, other=0.0)
    tmp30 = tl.load(in_ptr2 + (x0 + 64*((-16) + x1) + 512*x2), tmp28, other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.load(in_ptr3 + (x0 + 64*((-16) + x1) + 512*x2), tmp28, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr1 + (128 + x0 + 256*x2), tmp28, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = tmp33 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp28, tmp37, tmp38)
    tmp40 = tmp0 >= tmp26
    tmp41 = tl.full([1], 32, tl.int64)
    tmp42 = tmp0 < tmp41
    tmp43 = tl.load(in_ptr0 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp44 = tl.load(in_ptr2 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr3 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.load(in_ptr4 + (x0 + 64*((-24) + x1) + 512*x2), tmp40, other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tl.load(in_ptr1 + (192 + x0 + 256*x2), tmp40, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tmp49 * tmp51
    tmp53 = tmp49 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp40, tmp53, tmp54)
    tmp56 = tl.where(tmp28, tmp39, tmp55)
    tmp57 = tl.where(tmp15, tmp24, tmp56)
    tmp58 = tl.where(tmp4, tmp11, tmp57)
    tl.store(out_ptr0 + (x3), tmp58, None)
