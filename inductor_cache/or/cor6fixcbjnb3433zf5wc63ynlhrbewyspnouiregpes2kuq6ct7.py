
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x0 = (xindex % 4)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (4 + x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (x1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 24, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 24)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 24")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4), xmask)
    tmp8 = tl.full([XBLOCK], 7, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 7)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 7")
    tmp13 = tl.load(in_ptr2 + (x0 + 4*tmp11), xmask)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK], 32, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 32")
    tmp21 = tl.load(in_ptr3 + (x0 + 4*tmp19), xmask)
    tmp22 = tmp14 + tmp21
    tmp24 = tl.full([XBLOCK], 13, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 13)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 13")
    tmp29 = tl.load(in_ptr4 + (x0 + 4*tmp27), xmask)
    tmp30 = tmp22 + tmp29
    tmp31 = 0.0
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr0 + (x4), tmp32, xmask)
