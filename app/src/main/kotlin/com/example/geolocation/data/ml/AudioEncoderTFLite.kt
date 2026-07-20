package com.example.geolocation.data.ml

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import org.tensorflow.lite.Interpreter
import kotlin.math.min

/**
 * YAMNet mean-pool → [1024], parity with ml/encoders.py AudioEncoder.
 * Asset: `yamnet_meanpool.tflite` (float32 mono waveform or fixed log-mel — see export script).
 * Honest null when asset missing. Expects 16 kHz mono WAV when using raw waveform models.
 */
@Singleton
class AudioEncoderTFLite @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    companion object {
        const val ASSET_NAME = "yamnet_meanpool.tflite"
        const val DIM = 1024
        const val SAMPLE_RATE = 16_000
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

    fun embed(audio: File?): FloatArray? {
        if (audio == null || !audio.isFile) return null
        ensure()
        val interp = interpreter ?: return null
        return try {
            val waveform = readMonoPcm16OrFloatWav(audio) ?: return null
            // TFLite YAMNet exports often take [1, N] waveform
            val input = arrayOf(waveform)
            val out = Array(1) { FloatArray(DIM) }
            try {
                interp.run(input, out)
            } catch (_: Exception) {
                // Some exports use ByteBuffer input
                val bb = ByteBuffer.allocateDirect(4 * waveform.size).order(ByteOrder.nativeOrder())
                waveform.forEach { bb.putFloat(it) }
                bb.rewind()
                interp.run(bb, out)
            }
            out[0]
        } catch (e: Exception) {
            error = "audio embed failed: ${e.message}"
            null
        }
    }

    /** Minimal WAV reader: PCM 16-bit mono 16 kHz preferred; returns float [-1,1]. */
    private fun readMonoPcm16OrFloatWav(file: File): FloatArray? {
        val bytes = file.readBytes()
        if (bytes.size < 44) {
            // raw PCM int16 little-endian
            return pcm16ToFloat(bytes)
        }
        val riff = bytes.copyOfRange(0, 4).toString(Charsets.US_ASCII)
        if (riff != "RIFF") {
            return pcm16ToFloat(bytes)
        }
        // Find data chunk
        var i = 12
        var dataOffset = -1
        var dataSize = 0
        while (i + 8 <= bytes.size) {
            val id = bytes.copyOfRange(i, i + 4).toString(Charsets.US_ASCII)
            val size = (bytes[i + 4].toInt() and 0xff) or
                ((bytes[i + 5].toInt() and 0xff) shl 8) or
                ((bytes[i + 6].toInt() and 0xff) shl 16) or
                ((bytes[i + 7].toInt() and 0xff) shl 24)
            if (id == "data") {
                dataOffset = i + 8
                dataSize = size
                break
            }
            i += 8 + size + (size % 2)
        }
        if (dataOffset < 0) return null
        val end = min(bytes.size, dataOffset + dataSize)
        return pcm16ToFloat(bytes.copyOfRange(dataOffset, end))
    }

    private fun pcm16ToFloat(pcm: ByteArray): FloatArray {
        val n = pcm.size / 2
        if (n == 0) return FloatArray(0)
        val out = FloatArray(n)
        var j = 0
        for (i in 0 until n) {
            val lo = pcm[j].toInt() and 0xff
            val hi = pcm[j + 1].toInt()
            j += 2
            val s = (hi shl 8) or lo
            val signed = if (s > 32767) s - 65536 else s
            out[i] = (signed / 32768f).coerceIn(-1f, 1f)
        }
        return out
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
            error = "Asset $ASSET_NAME missing (${e.message}). Run ml/export_encoders_tflite.py"
            interpreter = null
        }
    }
}
