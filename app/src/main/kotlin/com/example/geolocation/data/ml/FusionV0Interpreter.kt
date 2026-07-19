package com.example.geolocation.data.ml

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import org.tensorflow.lite.Interpreter

/**
 * fusion_v0.tflite runner.
 *
 * Expected inputs (precomputed embeddings — not raw pixels/WAV):
 *   image_embedding [1,576], audio_embedding [1,1024], context [1,12], modality_mask [1,3]
 * Outputs:
 *   vibe_probabilities [1,7], perceptual_embedding [1,128], vibe_id [1]
 *
 * Honest unavailable when asset missing. Never invents random vibes.
 */
@Singleton
class FusionV0Interpreter @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    sealed class Result {
        data class Ok(
            val vibeLabel: String,
            val vibeId: Int,
            val probabilities: FloatArray,
            val perceptual: FloatArray,
        ) : Result()

        data class Unavailable(val reason: String) : Result()
    }

    private val labels = arrayOf(
        "serene", "energetic", "chaotic", "nostalgic", "tense", "social", "contemplative",
    )

    private var interpreter: Interpreter? = null
    private var attempted = false
    private var error: String? = null

    fun isAvailable(): Boolean {
        ensure()
        return interpreter != null
    }

    fun unavailableReason(): String? {
        ensure()
        return if (interpreter == null) error else null
    }

    fun run(
        imageEmbedding: FloatArray,
        audioEmbedding: FloatArray,
        context12: FloatArray,
        modalityMask: FloatArray,
    ): Result {
        ensure()
        val interp = interpreter
            ?: return Result.Unavailable(error ?: "fusion_v0.tflite not packaged")

        require(imageEmbedding.size == 576)
        require(audioEmbedding.size == 1024)
        require(context12.size == 12)
        require(modalityMask.size == 3)

        // Zero missing modalities
        val img = imageEmbedding.copyOf()
        val aud = audioEmbedding.copyOf()
        if (modalityMask[0] == 0f) img.fill(0f)
        if (modalityMask[1] == 0f) aud.fill(0f)

        return try {
            val inputs = arrayOf(
                arrayOf(img),
                arrayOf(aud),
                arrayOf(context12),
                arrayOf(modalityMask),
            )
            val probs = Array(1) { FloatArray(7) }
            val perc = Array(1) { FloatArray(128) }
            val outputs = hashMapOf<Int, Any>(
                0 to probs,
                1 to perc,
            )
            // Prefer signature API when model exported with signatures
            try {
                val inMap = hashMapOf<String, Any>(
                    "image_embedding" to arrayOf(img),
                    "audio_embedding" to arrayOf(aud),
                    "context" to arrayOf(context12),
                    "modality_mask" to arrayOf(modalityMask),
                )
                val outMap = hashMapOf<String, Any>(
                    "vibe_probabilities" to probs,
                    "perceptual_embedding" to perc,
                )
                interp.runSignature(inMap, outMap, "serving_default")
            } catch (_: Exception) {
                interp.runForMultipleInputsOutputs(inputs, outputs)
            }
            var best = 0
            for (i in 1 until 7) {
                if (probs[0][i] > probs[0][best]) best = i
            }
            Result.Ok(
                vibeLabel = labels[best],
                vibeId = best,
                probabilities = probs[0],
                perceptual = perc[0],
            )
        } catch (e: Exception) {
            Result.Unavailable("inference failed: ${e.message}")
        }
    }

    private fun ensure() {
        if (attempted) return
        attempted = true
        val asset = "fusion_v0.tflite"
        try {
            context.assets.open(asset).use { input ->
                val bytes = input.readBytes()
                val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                bb.put(bytes).rewind()
                interpreter = Interpreter(bb)
            }
        } catch (e: Exception) {
            error = "Asset $asset missing or unloadable (${e.message}). " +
                "Package a trained export — no untrained model is committed."
            interpreter = null
        }
    }
}
