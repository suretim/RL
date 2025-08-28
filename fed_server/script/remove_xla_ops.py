import tensorflow as tf
import os
#python remove_xla_ops.py
def list_ops_in_saved_model(saved_model_dir):
    """列出 SavedModel 中的所有 op 类型"""
    imported = tf.saved_model.load(saved_model_dir)
    concrete_func = imported.signatures["serving_default"]

    graph_def = concrete_func.graph.as_graph_def()
    ops = set()
    for node in graph_def.node:
        ops.add(node.op)
    return ops

def clean_saved_model(input_dir, output_dir):
    # 1. 检查原始模型是否包含 XLA 特有算子
    ops_before = list_ops_in_saved_model(input_dir)
    xla_ops = [op for op in ops_before if op.lower().startswith("xla")]
    print(f"📋 原模型包含 {len(ops_before)} 种算子")
    if xla_ops:
        print(f"⚠ 检测到 XLA 算子: {xla_ops}")
    else:
        print("✅ 未检测到 XLA 算子")

    # 2. 加载模型
    model = tf.saved_model.load(input_dir)
    infer = model.signatures["serving_default"]

    # 3. 重新构建无 XLA 的推理函数
    @tf.function(jit_compile=False)  # 禁用 XLA
    def no_xla_infer(*args, **kwargs):
        return infer(*args, **kwargs)

    # 4. 保存新模型
    tf.saved_model.save(
        obj=model,
        export_dir=output_dir,
        signatures={
            "serving_default": no_xla_infer.get_concrete_function(
                *infer.structured_input_signature[1].values()
            )
        }
    )
    print(f"✅ 已保存无 XLA 模型: {output_dir}")

    # 5. 检查新模型算子
    ops_after = list_ops_in_saved_model(output_dir)
    xla_ops_after = [op for op in ops_after if op.lower().startswith("xla")]
    print(f"📋 新模型包含 {len(ops_after)} 种算子")
    if xla_ops_after:
        print(f"❌ 清理后仍存在 XLA 算子: {xla_ops_after}")
    else:
        print("🎉 清理成功，已移除所有 XLA 算子")

if __name__ == "__main__":
    input_path = "saved_model"           # 原始模型目录
    output_path = "saved_model_no_xla"   # 清理后模型目录
    os.makedirs(output_path, exist_ok=True)
    clean_saved_model(input_path, output_path)
