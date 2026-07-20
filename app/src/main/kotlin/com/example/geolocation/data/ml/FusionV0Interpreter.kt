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
 *   image_embedding [1, IMAGE_DIM=576], audio_embedding [1, AUDIO_DIM=1024],
 *   context [1, 12], modality_mask [1, 3]
 * Outputs:
 *   vibe_probabilities [1, 7], perceptual_embedding [1, PERCEPTUAL_DIM=128]
 * Note: audio 1024 (YAMNet) is NOT E5 semantic 1024.
 *
 * Honest unavailable when asset missing. Never invents random vibes.
 */
@Singleton
class FusionV0Interpreter @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    companion object {
        const val ASSET_NAME = "fusion_v0.tflite"
        val LABELS = arrayOf(
            "serene", "energetic", "chaotic", "nostalgic", "tense", "social", "contemplative",
        )
    }

    sealed class Result {
        data class Ok(
            val vibeLabel: String,
            val vibeId: Int,
            val probabilities: FloatArray,
            val perceptual: FloatArray,
        ) : Result()

        data class Unavailable(val reason: String) : Result()
    }

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
            ?: return Result.Unavailable(error ?: "$ASSET_NAME not packaged")

        require(imageEmbedding.size == EmbeddingContract.IMAGE_DIM)
        require(audioEmbedding.size == EmbeddingContract.AUDIO_DIM)
        require(context12.size == EmbeddingContract.CONTEXT_DIM)
        require(modalityMask.size == EmbeddingContract.MASK_DIM)

        // Zero missing modalities (mask: photo, audio, time — time always 1)
        val img = imageEmbedding.copyOf()
        val aud = audioEmbedding.copyOf()
        if (modalityMask[0] == 0f) img.fill(0f)
        if (modalityMask[1] == 0f) aud.fill(0f)

        return try {
            val probs = Array(1) { FloatArray(EmbeddingContract.VIBE_PROBS_DIM) }
            val perc = Array(1) { FloatArray(EmbeddingContract.PERCEPTUAL_DIM) }
            // Prefer named signature outputs (never pick first dim-7 tensor by index).
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
                val inputs = arrayOf(
                    arrayOf(img),
                    arrayOf(aud),
                    arrayOf(context12),
                    arrayOf(modalityMask),
                )
                // Fallback: map by tensor name when available
                val outputs = LinkedHashMap<Int, Any>()
                for (i in 0 until interp.outputTensorCount) {
                    val name = interp.getOutputTensor(i).name() ?: ""
                    when {
                        name.contains("prob", ignoreCase = true) ||
                            name.contains("vibe_prob", ignoreCase = true) ->
                            outputs[i] = probs
                        name.contains("perceptual", ignoreCase = true) ||
                            name.contains("embedding", ignoreCase = true) ->
                            outputs[i] = perc
                    }
                }
                if (outputs.isEmpty()) {
                    outputs[0] = probs
                    outputs[1] = perc
                }
                interp.runForMultipleInputsOutputs(inputs, outputs)
            }
            var best = 0
            for (i in 1 until 7) {
                if (probs[0][i] > probs[0][best]) best = i
            }
            Result.Ok(
                vibeLabel = LABELS[best],
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
        try {
            context.assets.open(ASSET_NAME).use { input ->
                val bytes = input.readBytes()
                val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                bb.put(bytes).rewind()
                interpreter = Interpreter(bb)
            }
        } catch (e: Exception) {
            error = "Asset $ASSET_NAME missing or unloadable (${e.message}). " +
                "Package a trained export — no untrained model is committed."
            interpreter = null
        }
    }
}
