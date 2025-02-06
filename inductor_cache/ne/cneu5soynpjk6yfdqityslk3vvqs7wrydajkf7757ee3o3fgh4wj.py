
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_convolution_max_pool2d_with_indices_mean_relu_sub_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (512*((r2 % 64)) + 32768*((r2 + 128*x0 + 8192*x1) // 32768) + ((((r2 + 128*x0 + 8192*x1) // 64) % 512))), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((((r2 + 128*x0 + 8192*x1) // 64) % 512)), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (512*((r2 % 64)) + 32768*((r2 + 128*x0 + 8192*x1) // 32768) + ((((r2 + 128*x0 + 8192*x1) // 64) % 512))), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp8 = tmp4 - tmp7
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
