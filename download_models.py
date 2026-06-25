"""
双模型下载与 ONNX 导出
- CLIP ViT-B/32: 粗召回 (512维)
- ResNet50: 细粒度精排 (2048维)
输出到 models/ 目录
"""
import os
import sys
import torch
import torch.onnx
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def export_clip_onnx():
    """导出 OpenCLIP ViT-B/32 视觉编码器 → ONNX"""
    import open_clip

    print("=" * 55)
    print("  1/2 导出 CLIP ViT-B/32 视觉编码器")
    print("=" * 55)

    model_name = 'ViT-B-32'
    pretrained = 'laion2b_s34b_b79k'  # 最佳开源权重

    print(f"  模型: {model_name}")
    print(f"  权重: {pretrained}")
    print("  正在下载...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    visual = model.visual  # ViT 视觉编码器

    # 输出维度
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = visual(dummy)
        dim = out.shape[1]
    print(f"  输出维度: {dim}")

    # 导出 ONNX（使用 TorchScript 路径，避免 dynamo 编码问题）
    output_path = os.path.join(MODELS_DIR, 'clip_vit_b32_visual.onnx')
    print(f"  正在导出 ONNX → {output_path}")

    # 禁用 dynamo 导出，使用传统 TorchScript 路径
    import os as _os
    _os.environ['TORCH_LOGS'] = ''  # 抑制 verbose 输出

    torch.onnx.export(
        visual,
        dummy,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        dynamo=False,  # 使用传统 TorchScript 导出
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] 导出完成! 文件大小: {size_mb:.1f} MB, 维度: {dim}")

    # 同时保存预处理参数供后续使用
    # CLIP 的预处理: RGB, resize 224, center crop, normalize
    print(f"  CLIP 预处理: resize=224, center_crop, normalize(mean/std)")

    return dim


def export_resnet50_onnx():
    """导出 ResNet50 特征提取器 → ONNX"""
    import torchvision.models as models

    print()
    print("=" * 55)
    print("  2/2 导出 ResNet50 细粒度特征提取器")
    print("=" * 55)

    print("  正在加载预训练 ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    # 移除最后的 FC 分类层，保留 avgpool → 2048维特征
    class ResNet50FeatureExtractor(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                backbone.avgpool,
                torch.nn.Flatten(),
            )

        def forward(self, x):
            return self.backbone(x)

    feature_model = ResNet50FeatureExtractor(model)

    # 验证输出维度
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = feature_model(dummy)
        dim = out.shape[1]
    print(f"  输出维度: {dim}")

    # 导出 ONNX
    output_path = os.path.join(MODELS_DIR, 'resnet50_feature.onnx')
    print(f"  正在导出 ONNX → {output_path}")

    torch.onnx.export(
        feature_model,
        dummy,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        dynamo=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] 导出完成! 文件大小: {size_mb:.1f} MB, 维度: {dim}")

    return dim


def verify_models(clip_dim, resnet_dim):
    """验证导出的 ONNX 模型可用"""
    import onnxruntime as ort

    print()
    print("=" * 55)
    print("  验证 ONNX 模型")
    print("=" * 55)

    for name, path, expected_dim in [
        ('CLIP', os.path.join(MODELS_DIR, 'clip_vit_b32_visual.onnx'), clip_dim),
        ('ResNet50', os.path.join(MODELS_DIR, 'resnet50_feature.onnx'), resnet_dim),
    ]:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: inp})
        actual_dim = out[0].shape[1]
        assert actual_dim == expected_dim, f"{name}: 期望{expected_dim}, 实际{actual_dim}"
        print(f"  [OK] {name}: {actual_dim}维, 推理成功")


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)

    print()
    print("  双模型下载 & ONNX 导出")
    print(f"  输出目录: {MODELS_DIR}")
    print()

    clip_dim = export_clip_onnx()
    resnet_dim = export_resnet50_onnx()
    verify_models(clip_dim, resnet_dim)

    print()
    print("=" * 55)
    print("  全部完成!")
    print(f"  CLIP:     {clip_dim}维 (粗召回)")
    print(f"  ResNet50: {resnet_dim}维 (细粒度精排)")
    print("=" * 55)
