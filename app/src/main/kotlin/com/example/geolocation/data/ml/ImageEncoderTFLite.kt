package com.example.geolocation.data.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import org.tensorflow.lite.Interpreter

/**
 * MobileNetV3Small pool-avg → [576], parity with ml/encoders.py ImageEncoder.
 * Asset: `mobilenet_v3_small.tflite` (float32, 224×224×3, preprocessing in-graph or [0,255]).
 * Honest null when asset missing.
 */
@Singleton
class ImageEncoderTFLite @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    companion object {
        const val ASSET_NAME = "mobilenet_v3_small.tflite"
        const val DIM = EmbeddingContract.IMAGE_DIM
        const val SIDE = 224
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

    fun embed(photo: File?): FloatArray? {
        if (photo == null || !photo.isFile) return null
        ensure()
        val interp = interpreter ?: return null
        return try {
            val bmp = BitmapFactory.decodeFile(photo.absolutePath) ?: return null
            val input = preprocess(bmp)
            bmp.recycle()
            val out = Array(1) { FloatArray(DIM) }
            interp.run(input, out)
            out[0]
        } catch (e: Exception) {
            error = "image embed failed: ${e.message}"
            null
        }
    }

    /** Center-crop square, resize 224, float32 NHWC [0,255] for Keras include_preprocessing. */
    private fun preprocess(src: Bitmap): Array<Array<Array<FloatArray>>> {
        val w = src.width
        val h = src.height
        val side = minOf(w, h)
        val left = (w - side) / 2
        val top = (h - side) / 2
        val cropped = Bitmap.createBitmap(src, left, top, side, side)
        val resized = Bitmap.createScaledBitmap(cropped, SIDE, SIDE, true)
        if (cropped !== src) cropped.recycle()
        val tensor = Array(1) {
            Array(SIDE) { y ->
                Array(SIDE) { x ->
                    val p = resized.getPixel(x, y)
                    floatArrayOf(
                        ((p shr 16) and 0xff).toFloat(),
                        ((p shr 8) and 0xff).toFloat(),
                        (p and 0xff).toFloat(),
                    )
                }
            }
        }
        if (resized !== src) resized.recycle()
        return tensor
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
