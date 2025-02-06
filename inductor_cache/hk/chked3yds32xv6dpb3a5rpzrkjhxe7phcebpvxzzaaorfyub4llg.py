
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1024) % 128)
    x0 = (xindex % 1024)
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 64, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = x1
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.full([1], 32, tl.int64)
    tmp15 = tmp11 < tmp14
    tmp16 = tmp15 & tmp10
    tmp17 = tl.load(in_ptr0 + (x0 + 1024*(x1) + 32768*x2), tmp16, other=0.0)
    tmp18 = tl.load(in_ptr1 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp11 >= tmp14
    tmp23 = tl.full([1], 64, tl.int64)
    tmp24 = tmp11 < tmp23
    tmp25 = tmp22 & tmp10
    tmp26 = tl.load(in_ptr2 + (x0 + 1024*((-32) + (x1)) + 32768*x2), tmp25, other=0.0)
    tmp27 = tl.where(tmp15, tmp21, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp10, tmp27, tmp28)
    tmp30 = tmp5 >= tmp8
    tmp31 = tl.full([1], 96, tl.int64)
    tmp32 = tmp5 < tmp31
    tmp33 = tmp30 & tmp4
    tmp34 = tl.load(in_ptr3 + (x0 + 1024*((-64) + (x1)) + 32768*x2), tmp33, other=0.0)
    tmp35 = tl.where(tmp9, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp4, tmp35, tmp36)
    tmp38 = tmp0 >= tmp3
    tmp39 = tl.full([1], 128, tl.int64)
    tmp40 = tmp0 < tmp39
    tmp41 = tl.load(in_ptr4 + (x0 + 1024*((-96) + x1) + 32768*x2), tmp38, other=0.0)
    tmp42 = tl.where(tmp4, tmp37, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, None)
