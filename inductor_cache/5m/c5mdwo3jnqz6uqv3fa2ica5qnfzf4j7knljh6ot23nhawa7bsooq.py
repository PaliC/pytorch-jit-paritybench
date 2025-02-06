
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x6 = xindex
    x3 = ((xindex // 16) % 256)
    x4 = xindex // 4096
    x5 = (xindex % 4096)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x6), tmp8, None)
    tl.store(out_ptr1 + (x6), tmp25, None)
    tl.store(out_ptr2 + (x5 + 8192*x4), tmp8, None)
    tl.store(out_ptr3 + (x5 + 8704*x4), tmp8, None)
    tl.store(out_ptr4 + (x5 + 9216*x4), tmp8, None)
    tl.store(out_ptr5 + (x5 + 9728*x4), tmp8, None)
    tl.store(out_ptr6 + (x5 + 10240*x4), tmp8, None)
    tl.store(out_ptr7 + (x5 + 10752*x4), tmp8, None)
    tl.store(out_ptr8 + (x5 + 11264*x4), tmp8, None)
    tl.store(out_ptr9 + (x5 + 11776*x4), tmp8, None)
    tl.store(out_ptr10 + (x5 + 12288*x4), tmp8, None)
    tl.store(out_ptr11 + (x5 + 12800*x4), tmp8, None)
    tl.store(out_ptr12 + (x5 + 13312*x4), tmp8, None)
    tl.store(out_ptr13 + (x5 + 13824*x4), tmp8, None)
    tl.store(out_ptr14 + (x5 + 14336*x4), tmp8, None)
    tl.store(out_ptr15 + (x5 + 14848*x4), tmp8, None)
    tl.store(out_ptr16 + (x5 + 15360*x4), tmp8, None)
    tl.store(out_ptr17 + (x5 + 15872*x4), tmp8, None)
    tl.store(out_ptr18 + (x5 + 16384*x4), tmp8, None)
