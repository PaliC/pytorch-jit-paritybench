
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x3 = xindex // 64
    x1 = ((xindex // 64) % 16)
    x2 = xindex // 1024
    x4 = xindex
    tmp23 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 48, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (48*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 64, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (32*x1 + 1024*x2 + ((-48) + x0)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr2 + (16 + 32*x1 + 1024*x2 + ((-48) + x0)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp16 = tl.load(in_ptr2 + (512 + 32*x1 + 1024*x2 + ((-48) + x0)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = tl.load(in_ptr2 + (528 + 32*x1 + 1024*x2 + ((-48) + x0)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp9, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp39, None)
