package com.example.geolocation.data.ml

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import org.tensorflow.lite.Interpreter

/**
 * Optional LiteRT/TFLite adapter. Explicit unavailable when model asset missing.
 * No random fallback labels.
 */
@Singleton
class OnDeviceVibeAnalyzer @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    sealed class AnalysisResult {
        data class Available(
            val vibeLabel: String,
            val confidence: Float,
            val probs: FloatArray,
        ) : AnalysisResult()

        data class Unavailable(val reason: String) : AnalysisResult()
    }

    private val vibeLabels = arrayOf(
        "serene", "energetic", "chaotic", "nostalgic", "tense", "social", "contemplative",
    )

    private var interpreter: Interpreter? = null
    private var loadAttempted = false
    private var loadError: String? = null

    fun status(): AnalysisResult {
        ensureLoaded()
        return if (interpreter != null) {
            AnalysisResult.Available("serene", 0f, FloatArray(7))
        } else {
            AnalysisResult.Unavailable(loadError ?: "Model not loaded")
        }
    }

    /**
     * Feature path: precomputed embeddings would be filled by upstream extractors.
     * Without a packaged model file, returns Unavailable (honest).
     */
    fun analyze(
        imageEmb: FloatArray?,
        audioEmb: FloatArray?,
        contextFeatures: FloatArray?,
        hasImage: Boolean,
        hasAudio: Boolean,
        hasLocation: Boolean,
    ): AnalysisResult {
        ensureLoaded()
        val interp = interpreter
            ?: return AnalysisResult.Unavailable(loadError ?: "TFLite model unavailable")

        val img = imageEmb ?: FloatArray(576)
        val aud = audioEmb ?: FloatArray(1024)
        val ctx = contextFeatures ?: FloatArray(12)
        val mask = floatArrayOf(
            if (hasImage) 1f else 0f,
            if (hasAudio) 1f else 0f,
            if (hasLocation) 1f else 0f,
        )

        return try {
            val inputs = arrayOf(
                arrayOf(img),
                arrayOf(aud),
                arrayOf(ctx),
                arrayOf(mask),
            )
            // Output buffers depend on signature; try common flat maps
            val vibeOut = Array(1) { FloatArray(7) }
            val outputs = hashMapOf<Int, Any>(0 to vibeOut)
            interp.runForMultipleInputsOutputs(inputs, outputs)
            val probs = vibeOut[0]
            var best = 0
            for (i in 1 until probs.size) {
                if (probs[i] > probs[best]) best = i
            }
            AnalysisResult.Available(
                vibeLabel = vibeLabels.getOrElse(best) { "serene" },
                confidence = probs[best],
                probs = probs,
            )
        } catch (e: Exception) {
            AnalysisResult.Unavailable("Inference failed: ${e.message}")
        }
    }

    /** Convenience when only media paths known and no feature extractors yet. */
    fun analyzeMedia(photo: File?, audio: File?, hasLocation: Boolean): AnalysisResult {
        ensureLoaded()
        if (interpreter == null) {
            return AnalysisResult.Unavailable(loadError ?: "TFLite model unavailable")
        }
        // Without on-device embedding extractors, do not invent labels.
        return AnalysisResult.Unavailable(
            "Model present but embedding extractors not bundled; analysis unavailable",
        )
    }

    private fun ensureLoaded() {
        if (loadAttempted) return
        loadAttempted = true
        try {
            val assetName = "vibe-fusion-v0.tflite"
            val exists = try {
                context.assets.open(assetName).close()
                true
            } catch (_: Exception) {
                false
            }
            if (!exists) {
                loadError = "Asset $assetName not packaged (no untrained model committed)"
                return
            }
            context.assets.open(assetName).use { input ->
                val bytes = input.readBytes()
                val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                bb.put(bytes)
                bb.rewind()
                interpreter = Interpreter(bb)
            }
        } catch (e: Exception) {
            loadError = e.message ?: "TFLite load failed"
            interpreter = null
        }
    }
}
