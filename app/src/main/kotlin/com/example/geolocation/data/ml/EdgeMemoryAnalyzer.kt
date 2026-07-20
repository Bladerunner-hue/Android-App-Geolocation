package com.example.geolocation.data.ml

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Single edge-analysis port for capture telemetry.
 * fusion_v0 TFLite when packaged; honest Unavailable without extractors/model.
 */
interface EdgeMemoryAnalyzer {
    fun capability(): EdgeCapability
    fun analyze(input: MemoryAnalysisInput): EdgeAnalysisResult
}

data class MemoryAnalysisInput(
    val photo: File?,
    val audio: File?,
    val hasLocation: Boolean,
    val context12: FloatArray,
    val modalityMask: FloatArray,
    /** Precomputed embeddings when extractors exist; null → unavailable if model needs them. */
    val imageEmbedding: FloatArray? = null,
    val audioEmbedding: FloatArray? = null,
)

data class ModelIdentity(
    val modelId: String,
    val revision: String,
    val assetName: String,
    val contextRevision: String = ContextEncoderV1.REVISION,
)

data class EdgeCapability(
    val modelReady: Boolean,
    val extractorsReady: Boolean,
    val reasonIfUnavailable: String?,
)

sealed interface EdgeAnalysisResult {
    data class Success(
        val probabilities: FloatArray,
        val perceptualEmbedding: FloatArray,
        val vibeLabel: String,
        val confidence: Float,
        val model: ModelIdentity,
        val latencyMs: Long,
    ) : EdgeAnalysisResult

    data class Unavailable(val reason: String) : EdgeAnalysisResult
    data class Failed(val code: String, val detail: String? = null) : EdgeAnalysisResult
}

/**
 * Canonical adapter: one asset name (`fusion_v0.tflite`), one output contract.
 * Does not invent vibes. Does not claim Success when only "interpreter loaded".
 */
@Singleton
class FusionV0EdgeAnalyzer @Inject constructor(
    @ApplicationContext private val context: Context,
    private val fusion: FusionV0Interpreter,
) : EdgeMemoryAnalyzer {

    private val identity = ModelIdentity(
        modelId = "fusion_v0",
        revision = "r1",
        assetName = FusionV0Interpreter.ASSET_NAME,
        contextRevision = ContextEncoderV1.REVISION,
    )

    override fun capability(): EdgeCapability {
        val modelReady = fusion.isAvailable()
        // MobileNet / YAMNet not bundled yet — extractors always false on v0.
        val extractorsReady = false
        val reason = when {
            !modelReady -> fusion.unavailableReason() ?: "fusion_v0.tflite not packaged"
            !extractorsReady ->
                "Model present but MobileNet/YAMNet extractors not bundled; analysis unavailable"
            else -> null
        }
        return EdgeCapability(
            modelReady = modelReady,
            extractorsReady = extractorsReady,
            reasonIfUnavailable = reason,
        )
    }

    override fun analyze(input: MemoryAnalysisInput): EdgeAnalysisResult {
        val cap = capability()
        if (!cap.modelReady || !cap.extractorsReady) {
            return EdgeAnalysisResult.Unavailable(
                cap.reasonIfUnavailable ?: "edge analysis unavailable",
            )
        }
        val img = input.imageEmbedding
        val aud = input.audioEmbedding
        if (img == null || aud == null) {
            return EdgeAnalysisResult.Unavailable(
                "Precomputed embeddings required (extractors not on device yet)",
            )
        }
        if (input.modalityMask.size != 3) {
            return EdgeAnalysisResult.Failed("bad_mask", "modality_mask must be length 3")
        }
        val t0 = System.nanoTime()
        return when (
            val r = fusion.run(
                imageEmbedding = img,
                audioEmbedding = aud,
                context12 = input.context12,
                modalityMask = input.modalityMask,
            )
        ) {
            is FusionV0Interpreter.Result.Ok -> {
                val ms = (System.nanoTime() - t0) / 1_000_000L
                val best = r.probabilities.indices.maxByOrNull { r.probabilities[it] } ?: 0
                EdgeAnalysisResult.Success(
                    probabilities = r.probabilities,
                    perceptualEmbedding = r.perceptual,
                    vibeLabel = r.vibeLabel,
                    confidence = r.probabilities[best],
                    model = identity,
                    latencyMs = ms,
                )
            }
            is FusionV0Interpreter.Result.Unavailable ->
                EdgeAnalysisResult.Unavailable(r.reason)
        }
    }
}
