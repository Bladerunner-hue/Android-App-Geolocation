package com.example.geolocation

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented test for the GeoJournal package (device or emulator).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        // Debug builds append ".debug" to applicationId.
        assertTrue(
            "unexpected package: ${appContext.packageName}",
            appContext.packageName.startsWith("com.example.geolocation"),
        )
    }
}
