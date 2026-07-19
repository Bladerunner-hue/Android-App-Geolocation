package com.example.geolocation.data.ml

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import java.util.Calendar
import java.util.TimeZone

/**
 * Android parity for Python ml/context12.py (context12-v1).
 * Must stay bit-compatible with the training contract.
 */
object ContextEncoderV1 {
    const val REVISION = "context12-v1"
    const val DIM = 12

    /**
     * @param epochMillisUtc capture time UTC millis
     * @param utcOffsetMinutes timezone offset used for local hour/dow/doy
     */
    fun encode(
        epochMillisUtc: Long,
        utcOffsetMinutes: Int,
        latitude: Double?,
        longitude: Double?,
        accuracyM: Float?,
    ): FloatArray {
        val tz = TimeZone.getTimeZone("GMT")
        // Apply offset manually via calendar field
        val cal = Calendar.getInstance(tz)
        cal.timeInMillis = epochMillisUtc + utcOffsetMinutes * 60_000L

        val hour = cal.get(Calendar.HOUR_OF_DAY) +
            cal.get(Calendar.MINUTE) / 60.0 +
            cal.get(Calendar.SECOND) / 3600.0
        // Calendar.DAY_OF_WEEK: Sun=1 … Sat=7 → convert to Mon=0 … Sun=6
        val dowJava = cal.get(Calendar.DAY_OF_WEEK)
        val dow = ((dowJava + 5) % 7).toDouble()
        val doy = (cal.get(Calendar.DAY_OF_YEAR) - 1) + hour / 24.0

        val hasLocation = latitude != null && longitude != null
        val lat = if (hasLocation) {
            (latitude!! / 90.0).coerceIn(-1.0, 1.0)
        } else {
            0.0
        }
        val lonRad = if (hasLocation) Math.toRadians(longitude!!) else 0.0
        val accuracy = if (hasLocation) {
            min(
                ln(1.0 + max((accuracyM ?: 0f).toDouble(), 0.0)) / ln(1.0 + 5000.0),
                1.0,
            )
        } else {
            0.0
        }

        return floatArrayOf(
            sin(2 * PI * hour / 24).toFloat(),
            cos(2 * PI * hour / 24).toFloat(),
            sin(2 * PI * dow / 7).toFloat(),
            cos(2 * PI * dow / 7).toFloat(),
            sin(2 * PI * doy / 365.2425).toFloat(),
            cos(2 * PI * doy / 365.2425).toFloat(),
            (utcOffsetMinutes / 840.0).coerceIn(-1.0, 1.0).toFloat(),
            lat.toFloat(),
            if (hasLocation) sin(lonRad).toFloat() else 0f,
            if (hasLocation) cos(lonRad).toFloat() else 0f,
            accuracy.toFloat(),
            if (hasLocation) 1f else 0f,
        )
    }

    fun modalityMask(photoPresent: Boolean, audioPresent: Boolean): FloatArray =
        floatArrayOf(
            if (photoPresent) 1f else 0f,
            if (audioPresent) 1f else 0f,
            1f, // time always present
        )
}
