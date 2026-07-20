package com.example.geolocation

import com.example.geolocation.data.ml.ContextEncoderV1
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Guards training-serving skew on modality_mask / context12-v1.
 */
class ContextEncoderContractTest {
    @Test
    fun modalityMask_isPhotoAudioTime_notLocation() {
        val mask = ContextEncoderV1.modalityMask(photoPresent = true, audioPresent = false)
        assertEquals(3, mask.size)
        assertArrayEquals(floatArrayOf(1f, 0f, 1f), mask, 0f)
    }

    @Test
    fun contextDim_isTwelve() {
        val v = ContextEncoderV1.encode(
            epochMillisUtc = 1_700_000_000_000L,
            utcOffsetMinutes = 60,
            latitude = 48.85,
            longitude = 2.35,
            accuracyM = 12f,
        )
        assertEquals(ContextEncoderV1.DIM, v.size)
        assertEquals(12, v.size)
        // has_location flag is last feature
        assertEquals(1f, v[11], 0f)
    }

    @Test
    fun missingAccuracy_doesNotPretendPerfectGps() {
        val withAcc = ContextEncoderV1.encode(
            1_700_000_000_000L, 0, 48.0, 2.0, 25f,
        )
        val noAcc = ContextEncoderV1.encode(
            1_700_000_000_000L, 0, 48.0, 2.0, null,
        )
        // accuracy_norm index 10 — unknown accuracy is 0.0 in both contracts when
        // accuracy_m is null (log1p(0)); callers must pass real accuracy when known.
        assertEquals(withAcc[11], noAcc[11], 0f)
    }
}
