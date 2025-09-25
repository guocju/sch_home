import tvm022
from tvm022 import relax
from tvm022.script import relax as R, tir as T, ir as I
# from tvm018.ir.module import IRModule

@I.ir_module
class Test:
    @R.function
    def main(
        x: R.Tensor((1, 3, "H", "W"), "float32"),
    ) -> R.Tensor((1, 3, "H", "W"), "float32"):
        H = T.int64()
        W = T.int64()

        # 固定常量参数
        brightness_const = R.const(0.1, "float32")
        contrast_const = R.const(1.2, "float32")

        # 提升对比度: (x - 0.5) * contrast + 0.5
        y = R.subtract(x, R.const(0.5, "float32"))
        y = R.multiply(y, contrast_const)
        y = R.add(y, R.const(0.5, "float32"))

        # 调整亮度
        out = R.add(y, brightness_const)
        return out


if __name__ == "__main__":
    import numpy as np

    # 测试输入
    H, W = 224, 224
    img = np.random.rand(1, 3, H, W).astype('float32')

    # 构建并编译
    mod = Test
    ex = relax.build(mod, target="llvm")

    # 导出为共享库
    so_path = "image_enhance.so"
    ex.export_library(so_path)
    print(f"Saved shared library to {so_path}")

    # 加载并运行
    lib = tvm022.runtime.load_module(so_path)
    vm = relax.VirtualMachine(lib, tvm022.cpu())

    # 只传一个输入
    result = vm["main"](tvm022.nd.array(img))
    print(result.shape)  # (1, 3, H, W)
