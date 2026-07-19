package com.example.geolocation.data.remote.mapper

import com.example.geolocation.data.local.entity.MemoryEntity
import java.io.File
import java.time.Instant
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * Maps Room memory → multipart form for POST /api/memories/analyze.
 * Never invents tensors; only forwards what capture stored.
 */
object MemoryAnalyzeMapper {

    private fun text(s: String): RequestBody =
        s.toRequestBody("text/plain".toMediaTypeOrNull())

    fun floatArrayJson(json: String?): RequestBody? {
        if (json.isNullOrBlank()) return null
        return text(json)
    }

    fun vibesProbsFromEvidence(structuredJson: String?): RequestBody? {
        if (structuredJson.isNullOrBlank()) return null
        return try {
            val obj = JSONObject(structuredJson)
            if (!obj.has("vibe_probs")) return null
            text(obj.getJSONArray("vibe_probs").toString())
        } catch (_: Exception) {
            null
        }
    }

    data class AnalyzeParts(
        val clientUuid: RequestBody,
        val privateMode: RequestBody,
        val caption: RequestBody?,
        val latitude: RequestBody?,
        val longitude: RequestBody?,
        val capturedAt: RequestBody?,
        val onDeviceVibe: RequestBody?,
        val onDeviceConfidence: RequestBody?,
        val onDeviceProbs: RequestBody?,
        val perceptualEmbedding: RequestBody?,
        val insightEmbedding: RequestBody?,
        val modelVersion: RequestBody?,
        val analysisSource: RequestBody?,
        val structuredEvidence: RequestBody?,
        val requestEnrichment: RequestBody,
        val photo: MultipartBody.Part?,
        val audio: MultipartBody.Part?,
    )

    fun toParts(memory: MemoryEntity): AnalyzeParts {
        val photoPart = memory.photoPath?.let { path ->
            val f = File(path)
            if (!f.exists()) return@let null
            MultipartBody.Part.createFormData(
                "photo",
                f.name,
                f.asRequestBody("image/*".toMediaTypeOrNull()),
            )
        }
        val audioPart = memory.audioPath?.let { path ->
            val f = File(path)
            if (!f.exists()) return@let null
            MultipartBody.Part.createFormData(
                "audio",
                f.name,
                f.asRequestBody("audio/wav".toMediaTypeOrNull()),
            )
        }
        val capturedIso = try {
            Instant.ofEpochMilli(memory.capturedAtMs).toString()
        } catch (_: Exception) {
            null
        }
        return AnalyzeParts(
            clientUuid = text(memory.clientUuid),
            // Server hard-rejects true; we only sync when private is off.
            privateMode = text("false"),
            caption = memory.caption?.let { text(it) },
            latitude = memory.latitude?.toString()?.let { text(it) },
            longitude = memory.longitude?.toString()?.let { text(it) },
            capturedAt = capturedIso?.let { text(it) },
            onDeviceVibe = memory.vibeLabel?.let { text(it) },
            onDeviceConfidence = memory.vibeConfidence?.toString()?.let { text(it) },
            onDeviceProbs = vibesProbsFromEvidence(memory.structuredEvidenceJson),
            perceptualEmbedding = floatArrayJson(memory.perceptualEmbeddingJson),
            insightEmbedding = null,
            modelVersion = memory.modelVersion?.let { text(it) },
            analysisSource = text(memory.analysisSource),
            structuredEvidence = memory.structuredEvidenceJson?.let { text(it) },
            requestEnrichment = text(memory.enrichmentEnabled.toString()),
            photo = photoPart,
            audio = audioPart,
        )
    }

    fun floatsToJson(arr: FloatArray): String {
        val ja = JSONArray()
        arr.forEach { ja.put(it.toDouble()) }
        return ja.toString()
    }

    fun buildStructuredEvidence(
        vibeProbs: FloatArray?,
        context12: FloatArray?,
        modalityMask: FloatArray,
        source: String,
        contract: String = "fusion_v0",
        contextRevision: String = "context12-v1",
    ): String {
        val obj = JSONObject()
        obj.put("contract", contract)
        obj.put("context12_revision", contextRevision)
        obj.put("source", source)
        obj.put(
            "modality_mask",
            JSONArray().apply { modalityMask.forEach { put(it.toDouble()) } },
        )
        if (context12 != null) {
            obj.put(
                "context12",
                JSONArray().apply { context12.forEach { put(it.toDouble()) } },
            )
        }
        if (vibeProbs != null) {
            obj.put(
                "vibe_probs",
                JSONArray().apply { vibeProbs.forEach { put(it.toDouble()) } },
            )
        }
        return obj.toString()
    }
}
