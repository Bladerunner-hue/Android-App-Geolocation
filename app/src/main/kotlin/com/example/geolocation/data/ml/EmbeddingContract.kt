package com.example.geolocation.data.ml

/**
 * Single source of truth for tensor sizes across Android + Python + Postgres.
 *
 * Spaces are **not interchangeable**:
 * - fusion_v0 perceptual ≠ E5 semantic even if some dims coincide elsewhere
 * - YAMNet audio 1024 ≠ E5 text 1024
 */
object EmbeddingContract {
    /** MobileNetV3Small pool-avg */
    const val IMAGE_DIM = 576

    /** YAMNet mean-pool (audio encoder) — NOT E5 */
    const val AUDIO_DIM = 1024

    /** context12-v1 */
    const val CONTEXT_DIM = 12

    /** modality_mask: [photo, audio, time] */
    const val MASK_DIM = 3

    /** fusion_v0 vibe head */
    const val VIBE_PROBS_DIM = 7

    /** fusion_v0 perceptual head */
    const val PERCEPTUAL_DIM = 128

    /** Optional insight head */
    const val INSIGHT_DIM = 128

    /**
     * Caption/journal semantic search — direct E5 HTTP service.
     * Model: intfloat/e5-large-v2 @ http://127.0.0.1:6100
     * DB: memory_semantic_embeddings.embedding vector(1024)
     */
    const val SEMANTIC_DIM = 1024

    const val SEMANTIC_MODEL_ID = "intfloat/e5-large-v2"
    const val FUSION_MODEL_ID = "fusion_v0"
    const val CONTEXT_REVISION = "context12-v1"

    val VIBE_LABELS: List<String> = listOf(
        "serene", "energetic", "chaotic", "nostalgic", "tense", "social", "contemplative",
    )

    fun requireSemantic(vec: FloatArray, what: String = "semantic") {
        require(vec.size == SEMANTIC_DIM) {
            "$what dim ${vec.size} != SEMANTIC_DIM=$SEMANTIC_DIM (E5 e5-large-v2)"
        }
    }

    fun requirePerceptual(vec: FloatArray) {
        require(vec.size == PERCEPTUAL_DIM) {
            "perceptual dim ${vec.size} != PERCEPTUAL_DIM=$PERCEPTUAL_DIM (fusion_v0)"
        }
    }
}
