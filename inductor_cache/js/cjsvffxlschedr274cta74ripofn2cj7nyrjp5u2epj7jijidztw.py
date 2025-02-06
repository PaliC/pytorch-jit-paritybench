
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (3 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (2 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (1 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tl.full([XBLOCK], 24, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 24)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 24")
    tmp11 = tl.load(in_ptr3 + (x1 + 4*tmp9), xmask, eviction_policy='evict_last')
    tmp13 = tl.full([XBLOCK], 7, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 7)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 7")
    tmp18 = tl.load(in_ptr4 + (x1 + 4*tmp16), xmask, eviction_policy='evict_last')
    tmp19 = tmp11 + tmp18
    tmp21 = tl.full([XBLOCK], 32, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tl.device_assert(((0 <= tmp24) & (tmp24 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp24 < 32")
    tmp26 = tl.load(in_ptr5 + (x1 + 4*tmp24), xmask, eviction_policy='evict_last')
    tmp27 = tmp19 + tmp26
    tmp29 = tl.full([XBLOCK], 13, tl.int32)
    tmp30 = tmp28 + tmp29
    tmp31 = tmp28 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp28)
    tl.device_assert(((0 <= tmp32) & (tmp32 < 13)) | ~(xmask), "index out of bounds: 0 <= tmp32 < 13")
    tmp34 = tl.load(in_ptr6 + (x1 + 4*tmp32), xmask, eviction_policy='evict_last')
    tmp35 = tmp27 + tmp34
    tmp36 = 0.0
    tmp37 = tmp35 + tmp36
    tmp38 = tmp4 + tmp37
    tl.store(in_out_ptr0 + (x3), tmp38, xmask)
