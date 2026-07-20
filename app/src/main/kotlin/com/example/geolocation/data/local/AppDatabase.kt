package com.example.geolocation.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.example.geolocation.data.local.dao.MemoryDao
import com.example.geolocation.data.local.dao.MemoryTrainingLabelDao
import com.example.geolocation.data.local.dao.UserDao
import com.example.geolocation.data.local.entity.MemoryEntity
import com.example.geolocation.data.local.entity.MemoryTrainingLabelEntity
import com.example.geolocation.data.local.entity.UserEntity

@Database(
    entities = [
        UserEntity::class,
        MemoryEntity::class,
        MemoryTrainingLabelEntity::class,
    ],
    version = 6,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
    abstract fun memoryDao(): MemoryDao
    abstract fun memoryTrainingLabelDao(): MemoryTrainingLabelDao

    companion object {
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        clientUuid TEXT NOT NULL,
                        photoPath TEXT,
                        audioPath TEXT,
                        latitude REAL,
                        longitude REAL,
                        caption TEXT,
                        vibeLabel TEXT,
                        vibeConfidence REAL,
                        analysisStatus TEXT NOT NULL,
                        privateMode INTEGER NOT NULL,
                        cloudSyncEnabled INTEGER NOT NULL,
                        enrichmentEnabled INTEGER NOT NULL,
                        capturedAtMs INTEGER NOT NULL,
                        createdAtMs INTEGER NOT NULL,
                        syncStatus TEXT NOT NULL,
                        lastSyncError TEXT,
                        serverId INTEGER,
                        evidenceJson TEXT
                    )
                    """.trimIndent(),
                )
                db.execSQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS index_memories_clientUuid ON memories(clientUuid)",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_memories_capturedAtMs ON memories(capturedAtMs)",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_memories_syncStatus ON memories(syncStatus)",
                )
            }
        }

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS memory_training_labels (
                        id TEXT NOT NULL PRIMARY KEY,
                        memoryId TEXT NOT NULL,
                        sessionId TEXT NOT NULL,
                        primaryVibe TEXT NOT NULL,
                        secondaryVibesJson TEXT NOT NULL,
                        valence INTEGER,
                        arousal INTEGER,
                        labelConfidence INTEGER NOT NULL,
                        labelSource TEXT NOT NULL,
                        utcOffsetMinutes INTEGER NOT NULL,
                        locationAccuracyMeters REAL,
                        consentForTraining INTEGER NOT NULL,
                        consentForCloud INTEGER NOT NULL,
                        labelledAtEpochMillis INTEGER NOT NULL,
                        correctsLabelId TEXT
                    )
                    """.trimIndent(),
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_memory_training_labels_memoryId ON memory_training_labels(memoryId)",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_memory_training_labels_sessionId ON memory_training_labels(sessionId)",
                )
            }
        }

        /** v3 → v4: AI/ML alignment with backend 002 + label sync outbox. */
        val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE memories ADD COLUMN modelVersion TEXT")
                db.execSQL(
                    "ALTER TABLE memories ADD COLUMN analysisSource TEXT NOT NULL DEFAULT 'unavailable'",
                )
                db.execSQL("ALTER TABLE memories ADD COLUMN perceptualEmbeddingJson TEXT")
                db.execSQL("ALTER TABLE memories ADD COLUMN structuredEvidenceJson TEXT")
                db.execSQL(
                    "ALTER TABLE memory_training_labels ADD COLUMN syncStatus TEXT NOT NULL DEFAULT 'not_applicable'",
                )
                db.execSQL("ALTER TABLE memory_training_labels ADD COLUMN lastSyncError TEXT")
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_memory_training_labels_syncStatus " +
                        "ON memory_training_labels(syncStatus)",
                )
            }
        }

        /** v4 → v5: optional insight embedding column (server 002 parity). */
        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE memories ADD COLUMN insightEmbeddingJson TEXT")
            }
        }

        /** v5 → v6: E5 semantic 1024-D JSON (matches memory_semantic_embeddings). */
        val MIGRATION_5_6 = object : Migration(5, 6) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE memories ADD COLUMN semanticEmbeddingJson TEXT")
            }
        }
    }
}
