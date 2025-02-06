
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_29(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 16)
    x2 = xindex // 1024
    x3 = xindex // 64
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 16*(x0) + 512*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 32, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (256 + x1 + 16*((-16) + x0) + 512*x2), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr0 + (256 + x1 + 16*((-32) + x0) + 512*x2), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (16*x3 + ((-32) + x0)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tmp0 >= tmp12
    tmp23 = tl.full([1], 64, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (256 + x1 + 16*((-48) + x0) + 512*x2), tmp22, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (16*x3 + ((-48) + x0)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp22, tmp29, tmp30)
    tmp32 = tl.where(tmp14, tmp21, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tl.store(out_ptr0 + (x4), tmp34, None)
