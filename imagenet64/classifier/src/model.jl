# The classifier networks: `build_vgg` (below) and `build_resnet18` (further down), selected by
# name through `build_model`. The VGG design:
#
#   - 3×3, stride-1 convs keep full resolution inside a stage, so features are computed before any
#     spatial information is thrown away; two per stage gives a 5×5 receptive field per stage.
#   - Downsampling is a 2×2 max-pool between stages (keep the strongest response: the invariance a
#     classifier wants and an autoencoder must avoid).
#   - Width doubles as resolution halves, so every stage costs about the same FLOPs.
#   - Global average pooling, then dropout and one linear layer: a small head, so the conv stack has
#     to make the classes linearly separable.
#
#   64²×3 → [64, 64] → pool → 32²×64 → [128, 128] → pool → 16²×128 → [256, 256] → pool → 8²×256
#        → [512, 512] → global mean → 512 → dropout → Dense → nclass logits

using Lux

# conv_bn: conv without bias (BatchNorm's β replaces it), BatchNorm, ReLU.
conv_bn(cin, cout; stride = 1) =
    Chain(Conv((3, 3), cin => cout; stride, pad=1, use_bias=false), BatchNorm(cout), relu)
stage(cin, cout) = Chain(conv_bn(cin, cout), conv_bn(cout, cout))

# The head shared by both networks: global average pooling, dropout, one linear layer.
head(cin, nclass, dropout) = Chain(GlobalMeanPool(), FlattenLayer(), Dropout(dropout), Dense(cin => nclass))

# VGG-style: 8 convs, max-pool between stages. 5.2 M parameters.
function build_vgg(; nclass = 1000, dropout = 0.3f0)
    return Chain(
        stage(3, 64),    MaxPool((2, 2)),       # 64² → 32²
        stage(64, 128),  MaxPool((2, 2)),       # 32² → 16²
        stage(128, 256), MaxPool((2, 2)),       # 16² →  8²
        stage(256, 512),                        #  8²
        head(512, nclass, dropout),             # logits
    )
end

# ResNet basic block: y = relu(F(x) + S(x)), F = conv-BN-ReLU-conv-BN, S = identity, or a strided
# 1×1 conv + BN when the resolution or channel count changes. The identity path gives gradients a
# direct route back (∂y/∂x = I + ∂F/∂x) and lets a block start out as "do nothing".
function basic_block(cin, cout; stride = 1)
    body = Chain(conv_bn(cin, cout; stride),
                 Conv((3, 3), cout => cout; pad=1, use_bias=false), BatchNorm(cout))
    shortcut = (stride == 1 && cin == cout) ? NoOpLayer() :
               Chain(Conv((1, 1), cin => cout; stride, use_bias=false), BatchNorm(cout))
    return Chain(Parallel(+, body, shortcut), relu)
end

res_stage(cin, cout; stride) = Chain(basic_block(cin, cout; stride), basic_block(cout, cout))

# ResNet-18 for 64×64: a 3×3 stem at full resolution (no stride-2 stem or max-pool; 64 px is
# already small), 4 stages of 2 basic blocks, stride-2 convs to downsample, the same head as the
# VGG. 17 convs + the linear layer; ~11.2 M parameters.
#
#   64²×3 → stem → 64²×64 → [64] → 64² → [128]/2 → 32² → [256]/2 → 16² → [512]/2 → 8²×512 → head
function build_resnet18(; nclass = 1000, dropout = 0.3f0)
    return Chain(
        conv_bn(3, 64),                         # stem, 64²
        res_stage(64, 64;   stride = 1),        # 64²
        res_stage(64, 128;  stride = 2),        # 32²
        res_stage(128, 256; stride = 2),        # 16²
        res_stage(256, 512; stride = 2),        #  8²
        head(512, nclass, dropout),             # logits
    )
end

const MODELS = Dict("vgg" => build_vgg, "resnet18" => build_resnet18)
build_model(name = "vgg"; kwargs...) = MODELS[name](; kwargs...)
