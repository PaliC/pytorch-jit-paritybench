
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gather_linalg_vector_norm_sub_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gather_linalg_vector_norm_sub_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex // 16
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4 + 16*x2), xmask)
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 4")
    tmp11 = tl.load(in_ptr2 + (4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp13 = tmp11 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.load(in_ptr2 + (1 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (1 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp14 + tmp18
    tmp20 = tl.load(in_ptr2 + (2 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (2 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp22 = tmp20 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tmp19 + tmp23
    tmp25 = tl.load(in_ptr2 + (3 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (3 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tmp24 + tmp28
    tmp30 = tl.load(in_ptr3 + (4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp32 = tmp30 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.load(in_ptr3 + (1 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr3 + (1 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tmp33 + tmp37
    tmp39 = tl.load(in_ptr3 + (2 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr3 + (2 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp41 = tmp39 - tmp40
    tmp42 = tmp41 * tmp41
    tmp43 = tmp38 + tmp42
    tmp44 = tl.load(in_ptr3 + (3 + 4*tmp4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr3 + (3 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp46 = tmp44 - tmp45
    tmp47 = tmp46 * tmp46
    tmp48 = tmp43 + tmp47
    tl.store(out_ptr0 + (x5), tmp29, xmask)
    tl.store(out_ptr1 + (x5), tmp48, xmask)
