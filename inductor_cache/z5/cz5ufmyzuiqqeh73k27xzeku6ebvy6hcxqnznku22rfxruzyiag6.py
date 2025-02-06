
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 3)
    x2 = xindex // 192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1280*x1 + 4096*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp5 = tl.load(in_ptr0 + (192 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp7 = tl.load(in_ptr0 + (256 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp9 = tl.load(in_ptr0 + (320 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp11 = tl.load(in_ptr0 + (384 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp13 = tl.load(in_ptr0 + (448 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp15 = tl.load(in_ptr0 + (512 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp17 = tl.load(in_ptr0 + (576 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp19 = tl.load(in_ptr0 + (640 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp21 = tl.load(in_ptr0 + (704 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp23 = tl.load(in_ptr0 + (768 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp25 = tl.load(in_ptr0 + (832 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp27 = tl.load(in_ptr0 + (896 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp29 = tl.load(in_ptr0 + (960 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp31 = tl.load(in_ptr0 + (1024 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp33 = tl.load(in_ptr0 + (1088 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp35 = tl.load(in_ptr0 + (1152 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp37 = tl.load(in_ptr0 + (1216 + x0 + 1280*x1 + 4096*x2), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp32 = tmp31 + tmp30
    tmp34 = tmp33 + tmp32
    tmp36 = tmp35 + tmp34
    tmp38 = tmp37 + tmp36
    tmp39 = 0.05
    tmp40 = tmp38 * tmp39
    tl.store(out_ptr0 + (x3), tmp40, xmask)
