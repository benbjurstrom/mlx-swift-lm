//
//  Idefics3.swift
//  mlx-swift-lm
//
//  Created by SHUHONG WU on 12/13/24.
//

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration

public struct Idefics3Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public var numHiddenLayers: Int { _numHiddenLayers ?? 32 }
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let rmsNormEps: Float
        public let vocabSize: Int
        public let numKeyValueHeads: Int
        public let ropeTheta: Float
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }

        private let _numHiddenLayers: Int?
        private let _ropeTraditional: Bool?
        private let _tieWordEmbeddings: Bool?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case _numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabSize = "vocab_size"
            case numKeyValueHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public var numHiddenLayers: Int { _numHiddenLayers ?? 12 }
        public let hiddenSize: Int
        public var intermediateSize: Int { _intermediateSize ?? 3072 }
        public let numAttentionHeads: Int
        public let patchSize: Int
        public let imageSize: Int
        public var numChannels: Int { _numChannels ?? 3 }
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

        private let _numHiddenLayers: Int?
        private let _intermediateSize: Int?
        private let _numChannels: Int?
        private let _layerNormEps: Float?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _numHiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case _intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case _numChannels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration
    public let modelType: String
    public let ignoreIndex: Int
    public let vocabSize: Int
    public let scaleFactor: Int
    public let imageTokenId: Int
    public let imageTokenIndex: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case ignoreIndex = "ignore_index"
        case vocabSize = "vocab_size"
        case scaleFactor = "scale_factor"
        case imageTokenId = "image_token_id"
        case imageTokenIndex = "image_token_index"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.textConfig =
            try container
            .decode(TextConfiguration.self, forKey: .textConfig)
        self.visionConfig =
            try container
            .decode(VisionConfiguration.self, forKey: .visionConfig)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.ignoreIndex = (try? container.decode(Int.self, forKey: .ignoreIndex)) ?? -100
        self.vocabSize = (try? container.decode(Int.self, forKey: .vocabSize)) ?? 128259
        self.scaleFactor = (try? container.decode(Int.self, forKey: .scaleFactor)) ?? 2
        self.imageTokenId = (try? container.decode(Int.self, forKey: .imageTokenId)) ?? 49153
        self.imageTokenIndex =
            (try? container.decode(Int.self, forKey: .imageTokenIndex)) ?? self.imageTokenId
    }
}

// MARK: - Connector

private class Idefics3MLP: Module, UnaryLayer {
    @ModuleInfo var proj: Linear
    init(_ config: Idefics3Configuration) {
        let inputSize = config.visionConfig.hiddenSize * (config.scaleFactor * config.scaleFactor)
        let outputSize = config.textConfig.hiddenSize
        self._proj.wrappedValue = Linear(inputSize, outputSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = proj(x)
        return out
    }
}

private class Idefics3Connector: Module {
    let scaleFactor: Int
    @ModuleInfo(key: "modality_projection") var modalityProjection: Idefics3MLP

    init(_ config: Idefics3Configuration) {
        self.scaleFactor = config.scaleFactor
        self._modalityProjection.wrappedValue = Idefics3MLP(config)
    }

    func pixelShuffle(_ x: MLXArray, scaleFactor: Int) -> MLXArray {
        let B = x.dim(0)
        let seq = x.dim(1)
        let embed_dim = x.dim(2)
        let side = Int(Double(seq).squareRoot())

        var reshaped = x.reshaped(B, side, side, embed_dim)
        reshaped =
            reshaped
            .reshaped(B, side, side / scaleFactor, embed_dim * scaleFactor)
        reshaped = reshaped.transposed(0, 2, 1, 3)
        reshaped =
            reshaped
            .reshaped(
                B,
                side / scaleFactor,
                side / scaleFactor,
                embed_dim * (scaleFactor * scaleFactor)
            )
        reshaped = reshaped.transposed(0, 2, 1, 3)
        reshaped =
            reshaped
            .reshaped(
                B,
                seq / (scaleFactor * scaleFactor),
                embed_dim * (scaleFactor * scaleFactor)
            )
        return reshaped
    }

    func callAsFunction(_ imageHiddenStates: MLXArray) -> MLXArray {
        let shuffled = pixelShuffle(imageHiddenStates, scaleFactor: scaleFactor)
        let out = modalityProjection(shuffled)
        return out
    }
}

// MARK: - Language

private enum Language {
    fileprivate class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let scale: Float
        @ModuleInfo(key: "q_proj") var q_proj: Linear
        @ModuleInfo(key: "k_proj") var k_proj: Linear
        @ModuleInfo(key: "v_proj") var v_proj: Linear
        @ModuleInfo(key: "o_proj") var o_proj: Linear
        @ModuleInfo(key: "rope") var ropeEmbed: RoPE

        init(_ config: Idefics3Configuration.TextConfiguration) {
            let dim = config.hiddenSize
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads
            let headDim = dim / nHeads
            self.scale = pow(Float(headDim), -0.5)

            self._q_proj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._k_proj.wrappedValue = Linear(
                dim,
                nKVHeads * headDim,
                bias: false
            )
            self._v_proj.wrappedValue = Linear(
                dim,
                nKVHeads * headDim,
                bias: false
            )
            self._o_proj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            self._ropeEmbed.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache? = nil
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)
            var q = q_proj(x)
            var k = k_proj(x)
            var v = v_proj(x)

            q = q.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            q = ropeEmbed(q, offset: offset)
            k = ropeEmbed(k, offset: offset)

            let output = attentionWithCacheUpdate(
                queries: q,
                keys: k,
                values: v,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            let final = o_proj(output)
            return final
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate_proj: Linear
        @ModuleInfo(key: "down_proj") var down_proj: Linear
        @ModuleInfo(key: "up_proj") var up_proj: Linear
        init(dim: Int, hiddenDim: Int) {
            self._gate_proj.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down_proj.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._up_proj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let g = gate_proj(x)
            let r = down_proj(silu(g) * up_proj(x))
            return r
        }
    }

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(
            key: "post_attention_layernorm"
        ) var postAttentionLayerNorm: RMSNorm
        let mlp: MLP

        init(_ config: Idefics3Configuration.TextConfiguration) {
            self._selfAttn.wrappedValue = Attention(config)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            self.mlp = MLP(
                dim: config.hiddenSize,
                hiddenDim: config.intermediateSize
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let a = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + a
            let m = mlp(postAttentionLayerNorm(h))
            let out = h + m
            return out
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        var layers: [TransformerBlock]
        let norm: RMSNorm
        let config: Idefics3Configuration.TextConfiguration
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int] {
            (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        }
        var headDim: MLX.IntOrPair {
            .init(config.hiddenSize / config.numAttentionHeads)
        }

        init(_ config: Idefics3Configuration.TextConfiguration) {
            self.config = config
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize
            )
            self.layers = (0 ..< config.numHiddenLayers)
                .map { _ in TransformerBlock(config) }
            self.norm = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            let lmHeadNeeded = !config.tieWordEmbeddings
            if lmHeadNeeded {
                self._lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabSize,
                    bias: false
                )
            }
        }

        func getEmbeddings(for inputIds: MLXArray) -> MLXArray {
            let e = embedTokens(inputIds)
            return e
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputs_embeds: MLXArray? = nil
        ) -> LMOutput {
            let h: MLXArray
            if let inputs_embeds = inputs_embeds {
                h = inputs_embeds.asType(norm.weight.dtype)
            } else if let inputs = inputs {
                h = embedTokens(inputs)
            } else {
                fatalError(
                    "At least one of inputs or inputs_embeds must be provided."
                )
            }

            let mask = createAttentionMask(h: h, cache: cache?.first)
            var x = h
            for (i, layer) in layers.enumerated() {
                let c = i < (cache?.count ?? 0) ? cache![i] : nil
                x = layer(x, mask: mask, cache: c)
            }

            x = norm(x)
            let out = lmHead != nil ? lmHead!(x) : embedTokens.asLinear(x)
            return LMOutput(logits: out)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            // filter out rotary_emb.inv_freq
            return
                weights
                .filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        }
    }
}

// MARK: - Vision

private enum Vision {
    static func checkArrayShape(_ arr: MLXArray) -> Bool {
        if arr.ndim != 4 { return false }
        let (o, h, w, _) = (arr.dim(0), arr.dim(1), arr.dim(2), arr.dim(3))
        return (o >= h && o >= w && h == w)
    }

    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float
        @ModuleInfo(key: "q_proj") var q_proj: Linear
        @ModuleInfo(key: "k_proj") var k_proj: Linear
        @ModuleInfo(key: "v_proj") var v_proj: Linear
        @ModuleInfo(key: "out_proj") var o_proj: Linear

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.numHeads = config.numAttentionHeads
            let headDim = config.hiddenSize / config.numAttentionHeads
            self.scale = pow(Float(headDim), -0.5)
            self._q_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
            self._k_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
            self._v_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
            self._o_proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
            -> MLXArray
        {
            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            let q = q_proj(x).reshaped(B, L, numHeads, D / numHeads).transposed(
                0,
                2,
                1,
                3
            )
            let k = k_proj(x).reshaped(B, L, numHeads, D / numHeads).transposed(
                0,
                2,
                1,
                3
            )
            let v = v_proj(x).reshaped(B, L, numHeads, D / numHeads).transposed(
                0,
                2,
                1,
                3
            )

            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, D)
            let final = o_proj(output)
            return final
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .precise)

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.fc1 = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: true
            )
            self.fc2 = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let out = fc2(activation(fc1(x)))
            return out
        }
    }

    fileprivate class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var self_attn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self._self_attn.wrappedValue = Attention(config)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps
            )
            self.mlp = MLP(config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps
            )
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
            -> MLXArray
        {
            let h = x + self_attn(layerNorm1(x), mask: mask)
            let out = h + mlp(layerNorm2(h))
            return out
        }
    }

    fileprivate class Encoder: Module {
        var layers: [EncoderLayer]
        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.layers = (0 ..< config.numHiddenLayers)
                .map { _ in EncoderLayer(config) }
        }

        func callAsFunction(
            _ x: MLXArray, outputHiddenStates: Bool = false,
            mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        )
            -> (
                MLXArray,
                [MLXArray]?
            )
        {
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            var h = x
            for l in layers {
                h = l(h, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }
            return (h, encoderStates)
        }
    }

    fileprivate class VisionEmbeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let numPositions: Int

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize)
            )
            let numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            self.numPositions = numPatches
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches,
                dimensions: config.hiddenSize
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patchEmbeddings = patchEmbedding(x)
            patchEmbeddings = patchEmbeddings.flattened(start: 1, end: 2)
            let positionIds = MLXArray(0 ..< numPositions)[.newAxis, 0...]
            let posEmbedding = positionEmbedding(positionIds)
            let embeddings = patchEmbeddings + posEmbedding
            return embeddings
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "embeddings") var embeddings: VisionEmbeddings
        @ModuleInfo(key: "encoder") var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm
        let config: Idefics3Configuration.VisionConfiguration

        init(_ config: Idefics3Configuration.VisionConfiguration) {
            self.config = config
            self._embeddings.wrappedValue = VisionEmbeddings(config)
            self._encoder.wrappedValue = Encoder(config)
            self._postLayernorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps
            )
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = true) -> (
            MLXArray,
            MLXArray,
            [MLXArray]?
        ) {
            let e = embeddings(x)
            let (encoded, hiddenStates) = encoder(
                e,
                outputHiddenStates: outputHiddenStates
            )
            let pooler_output = postLayernorm(encoded)
            return (pooler_output, e, hiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()
            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else if k.contains("patch_embedding.weight") {
                    if Vision.checkArrayShape(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }
            return sanitizedWeights
        }
    }
}

// MARK: - Model

public class Idefics3: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: Vision.VisionModel
    @ModuleInfo(
        key: "language_model"
    ) private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "connector") private var connector: Idefics3Connector
    public let config: Idefics3Configuration

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var headDim: MLX.IntOrPair { languageModel.headDim }

    public var loraLayers: [Module] {
        languageModel.layers
    }

    public init(_ config: Idefics3Configuration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue =
            Language
            .LanguageModel(config.textConfig)
        self._connector.wrappedValue = Idefics3Connector(config)
    }

    private func getInputEmbeddings(inputIds: MLXArray?, pixelValues: MLXArray?) -> MLXArray {
        if pixelValues == nil {
            guard let inputIds = inputIds else {
                fatalError("inputIds required if no pixelValues")
            }
            let inputs_embeds = languageModel.getEmbeddings(for: inputIds)
            return inputs_embeds
        }

        guard let inputIds = inputIds, let pixelValues = pixelValues else {
            fatalError("inputIds and pixelValues required")
        }

        let inputs_embeds = languageModel.getEmbeddings(for: inputIds)
        let (pooler_output, _, _) = visionModel(
            pixelValues,
            outputHiddenStates: true
        )
        // Match dtype with inputs_embeds
        let image_features = connector(
            pooler_output.asType(inputs_embeds.dtype)
        )

        let final = prepareInputsForMultimodal(
            imageFeatures: image_features,
            inputs_embeds: inputs_embeds,
            inputIds: inputIds
        )
        return final
    }

    // inputs_merger
    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray, inputs_embeds: MLXArray, inputIds: MLXArray
    ) -> MLXArray {
        // Assumes bs == 1
        // inputIds shape: (1, seq_len)
        // asArray(Int.self) -> [[Int]], take [0] to get [Int]
        let ids: [[Int]] = [inputIds.asArray(Int.self)]

        let inputIdArray: [Int] = ids[0]

        let imageTokenIndex = config.imageTokenIndex
        let imagePositions = inputIdArray.enumerated().compactMap {
            $1 == imageTokenIndex ? $0 : nil
        }

        var segments = [MLXArray]()
        var start_idx = 0

        let chunkSize = imageFeatures.shape[1]  // 64
        let chunkCount = imagePositions.count / chunkSize  // Should be imageFeatures.shape[0]
        let chunks = (0 ..< chunkCount).map { startIndex in
            let start = startIndex * chunkSize
            let end = start + chunkSize
            return Array(imagePositions[start ..< end])
        }

        for (chunkIndex, chunk) in chunks.enumerated() {
            let currentImage = imageFeatures[chunkIndex]

            for (i, pos) in chunk.enumerated() {
                if pos > start_idx {
                    segments.append(inputs_embeds[0, start_idx ..< pos])
                }
                segments.append(currentImage[i ..< i + 1])
                start_idx = pos + 1
            }
        }

        if start_idx < inputs_embeds.dim(1) {
            segments.append(inputs_embeds[0, start_idx...])
        }

        let finalEmbeds = concatenated(segments, axis: 0)
        return finalEmbeds.expandedDimensions(axis: 0)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels
        let embeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues
        )
        let result = languageModel(nil, cache: cache, inputs_embeds: embeddings)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let out = languageModel(inputs, cache: cache).logits
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Rename keys to match Python logic
        var renamed = [String: MLXArray]()
        for (k, v) in weights {
            var newKey = k
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            } else if newKey.hasPrefix("lm_head.") {
                newKey = "language_model." + newKey
            }
            renamed[newKey] = v
        }

        var final = [String: MLXArray]()
        for (k, v) in renamed {
            if k.hasPrefix("text_model.") {
                let suffix = String(k.dropFirst("text_model.".count))
                final["language_model." + suffix] = v
            } else {
                final[k] = v
            }
        }

        // Remove rotary_emb.inv_freq
        final = final.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        return final
    }
}

// MARK: - Processor Configuration
public struct Idefics3ProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let longestEdge: Int
        enum CodingKeys: String, CodingKey {
            case longestEdge = "longest_edge"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let maxImageSize: Size?
    public let imageSequenceLength: Int?
    public let imageTokenId: Int?
    public let fakeImageTokenAroundImage: Int?

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case maxImageSize = "max_image_size"
        case imageSequenceLength = "image_seq_len"
        case imageTokenId = "image_token_id"
        case fakeImageTokenAroundImage = "fake_image_token_around_image"
    }
}

// MARK: - Message Generator

public struct Idefics3MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        // For Idefics3, images are represented as <image> in the content
        // The actual image token expansion happens in the processor
        var contentParts: [[String: Any]] = []

        // Add image placeholders first
        for _ in message.images {
            contentParts.append(["type": "image"])
        }

        // Add text content
        contentParts.append(["type": "text", "text": message.content])

        return [
            "role": message.role.rawValue,
            "content": contentParts,
        ]
    }
}

// MARK: - Processor

public class Idefics3Processor: UserInputProcessor {
    private let config: Idefics3ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let maxImageSize: Int  // Size of each tile for vision encoder (e.g., 512)
    private let longestEdge: Int   // Max size for initial resize (e.g., 2048)
    private let imageTokenId: Int
    private let fakeImageTokenAroundImage: Int?
    private let globalImageTagId: Int?
    private let imageSeqLen: Int

    // Token strings for building prompt
    private let imageToken = "<image>"
    private let fakeImageToken = "<fake_token_around_image>"
    private let globalImageToken = "<global-img>"

    /// Helper to look up a special token ID from the tokenizer
    private static func lookupTokenId(tokenizer: any Tokenizer, token: String) -> Int? {
        let encoded = tokenizer.encode(text: token)
        // If encoding returns exactly one token, that's our ID
        return encoded.count == 1 ? encoded[0] : nil
    }

    public init(
        _ config: Idefics3ProcessorConfiguration,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer

        // maxImageSize is the tile size for vision encoder (512 for Docling)
        self.maxImageSize = config.maxImageSize?.longestEdge ?? 364
        // longestEdge is the max size for initial resize (2048 for Docling)
        self.longestEdge = config.size.longestEdge
        // Image sequence length per tile (64 for Docling, 169 for standard)
        self.imageSeqLen = config.imageSequenceLength ?? 169

        // Look up special token IDs - try config first, then tokenizer, then defaults
        self.imageTokenId = config.imageTokenId
            ?? Self.lookupTokenId(tokenizer: tokenizer, token: "<image>")
            ?? 49153  // Standard Idefics3 default

        // fake_token_around_image for wrapping image sequences
        self.fakeImageTokenAroundImage = config.fakeImageTokenAroundImage
            ?? Self.lookupTokenId(tokenizer: tokenizer, token: "<fake_token_around_image>")

        // <global-img> token for marking global image region
        self.globalImageTagId = Self.lookupTokenId(tokenizer: tokenizer, token: "<global-img>")
    }

    /// Resize image to fit within longestEdge, then pad to make dimensions multiples of tileSize.
    /// This preserves aspect ratio exactly by adding black padding instead of stretching.
    private func resizeAndPadForTiling(_ image: CIImage, longestEdge: Int, tileSize: Int) -> CIImage {
        // Step 1: Resize to fit within longestEdge, preserving aspect ratio exactly
        let fitSize = MediaProcessing.bestFit(
            image.extent.size,
            in: CGSize(width: longestEdge, height: longestEdge)
        )
        let resizedImage = MediaProcessing.resampleBicubic(image, to: fitSize)

        // Step 2: Calculate padded dimensions (round up to multiples of tileSize)
        let paddedWidth = Int(ceil(fitSize.width / CGFloat(tileSize))) * tileSize
        let paddedHeight = Int(ceil(fitSize.height / CGFloat(tileSize))) * tileSize

        // Step 3: If already at multiples, no padding needed
        if Int(fitSize.width) == paddedWidth && Int(fitSize.height) == paddedHeight {
            return resizedImage
        }

        // Step 4: Create black background and center the image on it
        let padX = (CGFloat(paddedWidth) - fitSize.width) / 2
        let padY = (CGFloat(paddedHeight) - fitSize.height) / 2

        // Create black background
        let blackBackground = CIImage(color: .black).cropped(
            to: CGRect(x: 0, y: 0, width: paddedWidth, height: paddedHeight)
        )

        // Center the resized image on the background
        let centeredImage = resizedImage.transformed(
            by: CGAffineTransform(translationX: padX, y: padY)
        )

        // Composite image over black background
        return centeredImage.composited(over: blackBackground)
    }

    /// Split an image into tiles of maxImageSize, plus a global view.
    /// Returns (tiles, numRows, numCols) where tiles includes all patches + global at the end.
    /// Tiles are extracted in row-major order (top-to-bottom, left-to-right in visual space).
    /// Aspect ratio is preserved exactly by padding with black instead of stretching.
    private func splitImage(_ image: CIImage) -> ([CIImage], Int, Int) {
        let tileSize = maxImageSize

        // Resize and pad image so dimensions are exact multiples of tileSize
        // This preserves aspect ratio and ensures clean tile boundaries
        let paddedImage = resizeAndPadForTiling(image, longestEdge: longestEdge, tileSize: tileSize)

        let width = Int(paddedImage.extent.width)
        let height = Int(paddedImage.extent.height)

        // Calculate number of tiles
        let numCols = width / tileSize
        let numRows = height / tileSize

        // If image fits in a single tile, no splitting needed
        if numRows == 1 && numCols == 1 {
            return ([paddedImage], 0, 0)  // rows=0, cols=0 indicates single image
        }

        var tiles = [CIImage]()

        // Extract tiles in row-major order (top-to-bottom, left-to-right in visual space)
        // CIImage has y=0 at bottom, so we iterate rows in reverse to get visual top-to-bottom order
        for row in (0..<numRows).reversed() {
            for col in 0..<numCols {
                let x = col * tileSize
                let y = row * tileSize  // In CIImage coords, row 0 is at bottom

                let cropRect = CGRect(
                    x: CGFloat(x),
                    y: CGFloat(y),
                    width: CGFloat(tileSize),
                    height: CGFloat(tileSize)
                )

                var tile = paddedImage.cropped(to: cropRect)
                // Reset origin to (0,0)
                tile = tile.transformed(
                    by: CGAffineTransform(translationX: -tile.extent.origin.x, y: -tile.extent.origin.y))
                tiles.append(tile)
            }
        }

        // Add global view (full padded image resized to tileSize × tileSize)
        let globalView = MediaProcessing.resampleBicubic(
            paddedImage, to: CGSize(width: tileSize, height: tileSize))
        tiles.append(globalView)

        return (tiles, numRows, numCols)
    }

    /// Build image prompt STRING for split images (to replace <image> placeholder)
    /// Format: <fake><row_1_col_1><image>×N ... \n ... <fake><global-img><image>×N<fake>
    private func buildSplitImagePromptString(numRows: Int, numCols: Int) -> String {
        var result = ""
        let imageTokens = String(repeating: imageToken, count: imageSeqLen)

        // Add tokens for each tile with row/col markers
        for row in 1...numRows {
            for col in 1...numCols {
                result += fakeImageToken
                result += "<row_\(row)_col_\(col)>"
                result += imageTokens
            }
            // Add newline after each row
            result += "\n"
        }

        // Add extra newline before global image
        result += "\n"

        // Add global image tokens
        result += fakeImageToken
        result += globalImageToken
        result += imageTokens
        result += fakeImageToken

        return result
    }

    /// Build image prompt STRING for single (non-split) images
    /// Format: <fake><global-img><image>×N<fake>
    private func buildSingleImagePromptString() -> String {
        var result = ""
        result += fakeImageToken
        result += globalImageToken
        result += String(repeating: imageToken, count: imageSeqLen)
        result += fakeImageToken
        return result
    }

    public func prepare(input: UserInput) throws -> LMInput {
        // Generate messages using the message generator (handles proper structure with image placeholders)
        let messages = Idefics3MessageGenerator().generate(from: input)

        if input.images.isEmpty {
            // No image scenario - just apply chat template
            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        } else {
            // Single image scenario
            guard input.images.count == 1 else {
                throw VLMError.singleImageAllowed
            }

            // Step 1: Apply chat template to get formatted prompt with role markers
            // The chat template will include a single <image> placeholder that we'll expand
            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
            let decodedPrompt = tokenizer.decode(tokens: promptTokens, skipSpecialTokens: false)

            // Step 2: Load and preprocess image
            var image = try input.images[0].asCIImage()
            image = MediaProcessing.inSRGBToneCurveSpace(image)
            image = MediaProcessing.apply(image, processing: input.processing)

            // Step 3: Split image into tiles (splitImage handles resize with proper alignment)
            let (tiles, numRows, numCols) = splitImage(image)

            // Step 5: Build the expanded image prompt string based on whether image was split
            let imagePromptString: String
            if numRows == 0 && numCols == 0 {
                // Single image (no splitting)
                imagePromptString = buildSingleImagePromptString()
            } else {
                // Split image with row/col markers
                imagePromptString = buildSplitImagePromptString(numRows: numRows, numCols: numCols)
            }

            // Step 6: Replace <image> placeholder with expanded image tokens
            let expandedPrompt = decodedPrompt.replacingOccurrences(
                of: imageToken, with: imagePromptString)

            // Step 7: Re-encode the expanded prompt
            let finalTokens = tokenizer.encode(text: expandedPrompt)
            let promptArray = MLXArray(finalTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray)

            // Step 8: Process all tiles into pixel arrays
            var pixelArrays = [MLXArray]()
            for tile in tiles {
                // Normalize each tile
                let normalizedTile = MediaProcessing.normalize(
                    tile,
                    mean: config.imageMeanTuple,
                    std: config.imageStdTuple
                )
                var pixels = MediaProcessing.asMLXArray(normalizedTile)

                // asMLXArray returns [1, C, H, W], we need [H, W, C] for stacking
                // First squeeze the batch dimension, then transpose
                pixels = pixels.squeezed(axis: 0)  // [C, H, W]
                pixels = pixels.transposed(1, 2, 0)  // [H, W, C]
                pixelArrays.append(pixels)
            }

            // Stack all tiles into [numTiles, H, W, C]
            let stackedPixels = MLX.stacked(pixelArrays, axis: 0)

            return LMInput(
                text: .init(tokens: promptArray, mask: mask),
                image: .init(pixels: stackedPixels)
            )
        }
    }
}
