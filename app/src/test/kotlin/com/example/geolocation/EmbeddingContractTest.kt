package com.example.geolocation

import com.example.geolocation.data.ml.EmbeddingContract
import org.junit.Assert.assertEquals
import org.junit.Test

class EmbeddingContractTest {
    @Test
    fun semanticIsE5_1024_notFusion128() {
        assertEquals(1024, EmbeddingContract.SEMANTIC_DIM)
        assertEquals(128, EmbeddingContract.PERCEPTUAL_DIM)
        assertEquals("intfloat/e5-large-v2", EmbeddingContract.SEMANTIC_MODEL_ID)
    }

    @Test
    fun yamnetAndE5_sameWidth_differentSpaces() {
        // Width coincides; spaces must stay separate in code/docs.
        assertEquals(EmbeddingContract.AUDIO_DIM, EmbeddingContract.SEMANTIC_DIM)
        assertEquals(576, EmbeddingContract.IMAGE_DIM)
        assertEquals(7, EmbeddingContract.VIBE_LABELS.size)
    }
}
