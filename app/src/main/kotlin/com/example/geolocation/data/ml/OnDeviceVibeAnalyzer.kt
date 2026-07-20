package com.example.geolocation.data.ml

import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Legacy facade kept for call sites not yet migrated.
 * Delegates to [EdgeMemoryAnalyzer]. Prefer injecting EdgeMemoryAnalyzer directly.
 *
 * Important: [status] never returns a fake "serene" success when the model is merely loaded.
 */
@Singleton
class OnDeviceVibeAnalyzer @Inject constructor(
    private val edge: EdgeMemoryAnalyzer,
) {
    sealed class AnalysisResult {
        data class Available(
            val vibeLabel: String,
            val confidence: Float,
            val probs: FloatArray,
        ) : AnalysisResult()

        data class Unavailable(val reason: String) : AnalysisResult()
    }

    fun status(): AnalysisResult {
        val cap = edge.capability()
        return if (cap.modelReady && cap.extractorsReady) {
            // Ready to run when extractors exist — still not a prediction.
            AnalysisResult.Unavailable("Ready for inference (no status prediction without input)")
        } else {
            AnalysisResult.Unavailable(cap.reasonIfUnavailable ?: "unavailable")
        }
    }

    fun analyzeMedia(photo: File?, audio: File?, hasLocation: Boolean): AnalysisResult {
        val context12 = FloatArray(ContextEncoderV1.DIM)
        val mask = ContextEncoderV1.modalityMask(
            photoPresent = photo != null,
            audioPresent = audio != null,
        )
        return when (
            val r = edge.analyze(
                MemoryAnalysisInput(
                    photo = photo,
                    audio = audio,
                    hasLocation = hasLocation,
                    context12 = context12,
                    modalityMask = mask,
                ),
            )
        ) {
            is EdgeAnalysisResult.Success ->
                AnalysisResult.Available(r.vibeLabel, r.confidence, r.probabilities)
            is EdgeAnalysisResult.Unavailable ->
                AnalysisResult.Unavailable(r.reason)
            is EdgeAnalysisResult.Failed ->
                AnalysisResult.Unavailable("${r.code}: ${r.detail}")
        }
    }
}
