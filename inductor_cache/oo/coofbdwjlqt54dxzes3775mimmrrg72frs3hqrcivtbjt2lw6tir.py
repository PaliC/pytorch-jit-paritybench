
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp22 = tl.load(in_ptr5 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp28 = tl.load(in_ptr6 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp34 = tl.load(in_ptr7 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp39 = tl.load(in_ptr8 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp43 = tl.load(in_ptr9 + (x0), xmask)
    tmp44 = tl.load(in_ptr10 + (0))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp47 = tl.load(in_ptr11 + (0))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp54 = tl.load(in_ptr12 + (0))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp57 = tl.load(in_ptr13 + (0))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tmp24 = tl.full([XBLOCK], 1, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tmp30 = tmp29 + tmp24
    tmp31 = tmp29 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp29)
    tmp33 = tmp21 - tmp21
    tmp36 = tmp33 * tmp35
    tmp37 = tmp21 + tmp36
    tmp38 = tmp37 - tmp37
    tmp41 = tmp38 * tmp40
    tmp42 = tmp37 + tmp41
    tmp46 = tmp43 - tmp45
    tmp49 = tmp48 + tmp6
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tmp9 / tmp50
    tmp52 = tmp51 * tmp11
    tmp53 = tmp46 * tmp52
    tmp56 = tmp53 * tmp55
    tmp59 = tmp56 + tmp58
    tmp60 = triton_helpers.maximum(tmp20, tmp59)
    tmp61 = tmp42 + tmp60
    tl.store(in_out_ptr0 + (x0), tmp61, xmask)
