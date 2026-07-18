package com.example.geolocation.data.ml

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Mono 16 kHz PCM16 WAV, hard-capped at 10 seconds.
 */
@Singleton
class AmbientAudioRecorder @Inject constructor() {

    data class Result(val file: File, val durationMs: Long)

    suspend fun recordToFile(
        outFile: File,
        maxDurationMs: Long = MAX_DURATION_MS,
    ): Result? = withContext(Dispatchers.IO) {
        val sampleRate = SAMPLE_RATE
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        if (minBuf <= 0) return@withContext null

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            minBuf * 2,
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            return@withContext null
        }

        outFile.parentFile?.mkdirs()
        val pcmTmp = File(outFile.parentFile, outFile.nameWithoutExtension + ".pcm")
        val started = System.currentTimeMillis()
        try {
            FileOutputStream(pcmTmp).use { fos ->
                val buf = ByteArray(minBuf)
                recorder.startRecording()
                withTimeoutOrNull(maxDurationMs) {
                    while (isActive) {
                        val n = recorder.read(buf, 0, buf.size)
                        if (n > 0) fos.write(buf, 0, n)
                    }
                }
            }
        } finally {
            try {
                recorder.stop()
            } catch (_: Exception) {
            }
            recorder.release()
        }
        val duration = System.currentTimeMillis() - started
        writeWavHeader(pcmTmp, outFile, sampleRate, 1, 16)
        pcmTmp.delete()
        Result(outFile, duration.coerceAtMost(maxDurationMs))
    }

    private fun writeWavHeader(
        pcm: File,
        wav: File,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int,
    ) {
        val pcmData = pcm.readBytes()
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(36 + pcmData.size)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1) // PCM
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort((channels * bitsPerSample / 8).toShort())
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray())
        header.putInt(pcmData.size)
        FileOutputStream(wav).use { out ->
            out.write(header.array())
            out.write(pcmData)
        }
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val MAX_DURATION_MS = 10_000L
    }
}
