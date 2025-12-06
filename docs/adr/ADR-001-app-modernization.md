# ADR-001: Modernization of Android Geolocation Application

**Status:** Proposed  
**Date:** 2025-01-15  
**Decision Makers:** Development Team  
**Technical Story:** Upgrade legacy Java-based Android application to modern standards with PostgreSQL backend integration

---

## Context and Problem Statement

The current Android application (`com.example.school`) is built using outdated patterns and technologies:

### Current State Analysis

| Component | Current | Issues |
|-----------|---------|--------|
| Language | Java | Not Kotlin-first (Google's recommendation since 2019) |
| UI Framework | XML Layouts + Activities | Not declarative, harder to maintain |
| Architecture | No clear pattern | Code in Activities, no separation of concerns |
| Async Operations | `AsyncTask` (deprecated) | Memory leaks, lifecycle issues |
| Backend | Simulated (`Thread.sleep()`) | No real data persistence |
| Target SDK | 33 | Should upgrade to 34+ |
| Play Services Location | 16.0.0 | Outdated, current is 21.x |

### Current Code Issues (from analysis)

1. **ActivityB.java** uses deprecated `AsyncTask`:
   ```java
   private class FetchDataTask extends AsyncTask<Void, Void, String> {
       // Deprecated API, causes memory leaks
   }
   ```

2. **No MVVM/Clean Architecture** - Business logic mixed with UI code

3. **No Repository Pattern** - Direct simulated data access in Activities

4. **No Dependency Injection** - Hard-coded dependencies

### PostgreSQL Database Schema (EffuzionBridge)

The existing PostgreSQL database has a `users` table ready for authentication:

| Column | Type | Purpose |
|--------|------|---------|
| id | integer | Primary key |
| username | varchar | User login name |
| email | varchar | User email |
| password_hash | varchar | Bcrypt hashed password |
| api_key | varchar | API authentication token |
| is_admin | boolean | Admin privileges |
| is_active | boolean | Account status |
| created_at | timestamp | Registration date |
| last_login | timestamp | Last login timestamp |

---

## Decision Drivers

1. **Maintainability** - Current code is hard to test and extend
2. **Modern UX** - Users expect Material Design 3 aesthetics
3. **Performance** - Kotlin coroutines vs deprecated AsyncTask
4. **Security** - Proper authentication with PostgreSQL backend
5. **Scalability** - Clean architecture enables feature growth
6. **Developer Experience** - Kotlin reduces boilerplate by ~40%

---

## Considered Options

### Option 1: Incremental Java Refactoring
- Keep Java, add ViewModel/LiveData
- Pros: Lower risk, smaller learning curve
- Cons: Doesn't address language limitations, no Compose

### Option 2: Full Kotlin + Compose Migration (Recommended)
- Complete rewrite with modern stack
- Pros: Best practices, modern UI, full benefits
- Cons: Higher initial effort, team training needed

### Option 3: Kotlin Migration Without Compose
- Convert to Kotlin, keep XML layouts
- Pros: Moderate effort, interoperability
- Cons: Misses declarative UI benefits

---

## Decision Outcome

**Chosen Option: Option 2 - Full Kotlin + Compose Migration**

This decision aligns with Google's Kotlin-first approach and provides the best long-term maintainability.

---

## Technical Architecture

### 1. Recommended Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI LAYER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Compose Screens │  │  ViewModels     │  │   UI State      │  │
│  │ (Material 3)    │  │  (StateFlow)    │  │   (Data Class)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       DOMAIN LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Use Cases     │  │   Entities      │  │   Interfaces    │  │
│  │ (Business Logic)│  │   (Models)      │  │   (Contracts)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Repositories   │  │  Remote Source  │  │  Local Source   │  │
│  │ (Data Mediator) │  │  (Retrofit API) │  │  (Room Cache)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (New)                             │
│               REST API → PostgreSQL (EffuzionBridge)             │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Kotlin 1.9+ | Primary development language |
| **UI** | Jetpack Compose + Material 3 | Declarative, modern UI |
| **Architecture** | MVVM + Clean Architecture | Separation of concerns |
| **DI** | Hilt (Dagger) | Dependency injection |
| **Async** | Kotlin Coroutines + Flow | Reactive data streams |
| **Network** | Retrofit + OkHttp + Moshi | REST API communication |
| **Local DB** | Room | SQLite abstraction, caching |
| **Navigation** | Navigation Compose | Type-safe navigation |
| **Location** | FusedLocationProviderClient (latest) | Geolocation services |
| **Auth** | JWT Tokens | Secure authentication |
| **Testing** | JUnit5, Mockk, Compose Testing | Unit & UI tests |

### 3. Gradle Configuration Changes

#### Root `build.gradle` (updated)
```kotlin
plugins {
    id 'com.android.application' version '8.2.0' apply false
    id 'org.jetbrains.kotlin.android' version '1.9.21' apply false
    id 'com.google.dagger.hilt.android' version '2.48' apply false
    id 'org.jetbrains.kotlin.plugin.compose' version '1.9.21' apply false
}
```

#### App `build.gradle` (updated)
```kotlin
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
    id 'org.jetbrains.kotlin.plugin.compose'
    id 'com.google.dagger.hilt.android'
    id 'kotlin-kapt'
}

android {
    namespace 'com.example.geolocation'
    compileSdk 34

    defaultConfig {
        applicationId "com.example.geolocation"
        minSdk 24  // Updated for Compose
        targetSdk 34
        versionCode 2
        versionName "2.0.0"
    }

    buildFeatures {
        compose true
    }

    composeOptions {
        kotlinCompilerExtensionVersion '1.5.7'
    }

    kotlinOptions {
        jvmTarget = '17'
    }
}

dependencies {
    // Compose BOM
    def composeBom = platform('androidx.compose:compose-bom:2024.02.00')
    implementation composeBom

    // Compose
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.ui:ui-graphics'
    implementation 'androidx.compose.ui:ui-tooling-preview'
    implementation 'androidx.compose.material3:material3'
    implementation 'androidx.activity:activity-compose:1.8.2'
    implementation 'androidx.navigation:navigation-compose:2.7.6'
    implementation 'androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0'
    implementation 'androidx.lifecycle:lifecycle-runtime-compose:2.7.0'

    // Hilt
    implementation 'com.google.dagger:hilt-android:2.48'
    kapt 'com.google.dagger:hilt-compiler:2.48'
    implementation 'androidx.hilt:hilt-navigation-compose:1.1.0'

    // Retrofit + Moshi
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-moshi:2.9.0'
    implementation 'com.squareup.moshi:moshi-kotlin:1.15.0'
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.12.0'

    // Room
    implementation 'androidx.room:room-runtime:2.6.1'
    implementation 'androidx.room:room-ktx:2.6.1'
    kapt 'androidx.room:room-compiler:2.6.1'

    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'

    // Location
    implementation 'com.google.android.gms:play-services-location:21.1.0'

    // DataStore (for preferences/tokens)
    implementation 'androidx.datastore:datastore-preferences:1.0.0'

    // Testing
    testImplementation 'junit:junit:4.13.2'
    testImplementation 'io.mockk:mockk:1.13.9'
    androidTestImplementation composeBom
    androidTestImplementation 'androidx.compose.ui:ui-test-junit4'
    debugImplementation 'androidx.compose.ui:ui-tooling'
    debugImplementation 'androidx.compose.ui:ui-test-manifest'
}
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
- [ ] Set up Kotlin project structure
- [ ] Configure Hilt dependency injection
- [ ] Create base architecture classes
- [ ] Set up Compose theming (Material 3)
- [ ] Create Backend REST API (Python/FastAPI or Node.js)

### Phase 2: Authentication (Week 3)
- [ ] Implement login screen (Compose)
- [ ] Create AuthRepository + AuthViewModel
- [ ] JWT token storage (DataStore)
- [ ] API integration with PostgreSQL users table
- [ ] Session management

### Phase 3: Core Features (Week 4-5)
- [ ] Migrate ActivityA → HomeScreen
- [ ] Migrate ActivityB → DashboardScreen
- [ ] Migrate ActivityC → LocationScreen
- [ ] Implement location tracking with modern API
- [ ] Camera integration with CameraX

### Phase 4: Polish & Testing (Week 6)
- [ ] Unit tests for ViewModels and Repositories
- [ ] UI tests with Compose Testing
- [ ] Error handling and offline support
- [ ] Performance optimization
- [ ] Documentation

---

## New Package Structure

```
com.example.geolocation/
├── di/                          # Hilt modules
│   ├── AppModule.kt
│   ├── NetworkModule.kt
│   └── DatabaseModule.kt
├── data/
│   ├── local/
│   │   ├── dao/
│   │   │   └── UserDao.kt
│   │   ├── entity/
│   │   │   └── UserEntity.kt
│   │   └── AppDatabase.kt
│   ├── remote/
│   │   ├── api/
│   │   │   └── AuthApi.kt
│   │   ├── dto/
│   │   │   ├── LoginRequest.kt
│   │   │   └── LoginResponse.kt
│   │   └── interceptor/
│   │       └── AuthInterceptor.kt
│   └── repository/
│       ├── AuthRepository.kt
│       └── LocationRepository.kt
├── domain/
│   ├── model/
│   │   ├── User.kt
│   │   └── Location.kt
│   └── usecase/
│       ├── LoginUseCase.kt
│       └── GetLocationUseCase.kt
├── ui/
│   ├── theme/
│   │   ├── Color.kt
│   │   ├── Theme.kt
│   │   └── Type.kt
│   ├── navigation/
│   │   └── NavGraph.kt
│   ├── screen/
│   │   ├── auth/
│   │   │   ├── LoginScreen.kt
│   │   │   └── LoginViewModel.kt
│   │   ├── home/
│   │   │   ├── HomeScreen.kt
│   │   │   └── HomeViewModel.kt
│   │   ├── dashboard/
│   │   │   ├── DashboardScreen.kt
│   │   │   └── DashboardViewModel.kt
│   │   └── location/
│   │       ├── LocationScreen.kt
│   │       └── LocationViewModel.kt
│   └── component/
│       ├── AppButton.kt
│       ├── LoadingIndicator.kt
│       └── ErrorDialog.kt
├── util/
│   ├── Result.kt
│   └── Extensions.kt
└── GeolocationApp.kt            # Hilt Application class
```

---

## Backend API Design

A new REST API is required to connect the Android app to PostgreSQL:

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Authenticate user |
| POST | `/api/auth/register` | Create new user |
| POST | `/api/auth/refresh` | Refresh JWT token |
| GET | `/api/user/profile` | Get current user |
| PUT | `/api/user/profile` | Update profile |
| POST | `/api/location` | Save location data |
| GET | `/api/location/history` | Get location history |

### Example Login Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Android   │────►│   REST API  │────►│  PostgreSQL │
│     App     │     │  (FastAPI)  │     │ EffuzionBridge│
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │
      │ POST /auth/login   │                    │
      │ {username, pass}   │                    │
      │───────────────────►│                    │
      │                    │ SELECT * FROM users│
      │                    │ WHERE username=?   │
      │                    │───────────────────►│
      │                    │                    │
      │                    │◄──────────────────│
      │                    │   user record      │
      │                    │                    │
      │                    │ verify password_hash│
      │                    │ generate JWT       │
      │◄───────────────────│                    │
      │  {token, user}     │                    │
      │                    │                    │
      │ Store token in     │                    │
      │ DataStore          │                    │
```

---

## UI Mockups (Compose Screens)

### Login Screen
```
┌────────────────────────────┐
│        🌍 GeoTrack         │
│                            │
│  ┌──────────────────────┐  │
│  │ Username             │  │
│  └──────────────────────┘  │
│                            │
│  ┌──────────────────────┐  │
│  │ Password         👁   │  │
│  └──────────────────────┘  │
│                            │
│  ┌──────────────────────┐  │
│  │       LOGIN          │  │
│  └──────────────────────┘  │
│                            │
│     Don't have account?    │
│        Register here       │
└────────────────────────────┘
```

### Dashboard Screen (Material 3)
```
┌────────────────────────────┐
│ ≡  Dashboard         👤    │
├────────────────────────────┤
│                            │
│  ┌────────┐  ┌────────┐   │
│  │   📍   │  │   📷   │   │
│  │Location│  │ Camera │   │
│  └────────┘  └────────┘   │
│                            │
│  ┌────────┐  ┌────────┐   │
│  │   📊   │  │   ⚙️   │   │
│  │ History│  │Settings│   │
│  └────────┘  └────────┘   │
│                            │
│  Recent Locations          │
│  ├─ 40.7128°N, 74.0060°W  │
│  │  New York • 2 min ago   │
│  └─ 34.0522°N, 118.2437°W │
│     Los Angeles • 1 hr ago │
└────────────────────────────┘
```

---

## Migration Mapping

| Current (Java) | New (Kotlin + Compose) |
|----------------|------------------------|
| `ActivityA.java` | `HomeScreen.kt` + `HomeViewModel.kt` |
| `ActivityB.java` | `DashboardScreen.kt` + `DashboardViewModel.kt` |
| `ActivityC.java` | `LocationScreen.kt` + `LocationViewModel.kt` |
| `ActivityCi.java` | `LocationViewModel.kt` (merged) |
| XML layouts | Compose `@Composable` functions |
| `AsyncTask` | Kotlin Coroutines (`viewModelScope.launch`) |
| `SharedPreferences` | DataStore |
| Direct location calls | `LocationRepository` |

---

## Security Considerations

1. **Password Storage**: Continue using bcrypt hashes (already in DB)
2. **API Authentication**: JWT with refresh tokens
3. **Token Storage**: Android EncryptedSharedPreferences or DataStore
4. **HTTPS**: Enforce TLS for all API calls
5. **Certificate Pinning**: Optional for production
6. **Proguard/R8**: Enable code obfuscation

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Team unfamiliar with Kotlin/Compose | Medium | High | Training sessions, pair programming |
| Backend API development delays | Medium | Medium | Start backend work in Phase 1 |
| Location API breaking changes | Low | Medium | Abstract behind repository |
| Performance issues with Compose | Low | Low | Profile with Android Studio |

---

## Success Metrics

- [ ] 100% Kotlin codebase (0 Java files)
- [ ] All screens using Jetpack Compose
- [ ] Unit test coverage > 70%
- [ ] Successful login/logout with PostgreSQL
- [ ] Location tracking functional
- [ ] App size < 10MB
- [ ] Crash-free rate > 99%

---

## References

1. [Android App Architecture Guide](https://developer.android.com/topic/architecture)
2. [Jetpack Compose Setup](https://developer.android.com/develop/ui/compose/setup)
3. [Kotlin-First Android Development](https://developer.android.com/kotlin/first)
4. [Hilt Dependency Injection](https://developer.android.com/training/dependency-injection/hilt-android)
5. [Material Design 3](https://m3.material.io/)
6. [Android Architecture Samples](https://github.com/android/architecture-samples)

---

## Decision

**Approved / Rejected / Deferred**: _Pending Review_

**Rationale**: This modernization addresses all identified technical debt while positioning the application for future growth. The PostgreSQL backend integration enables real user authentication and data persistence.

---

*Document Version: 1.0*  
*Last Updated: 2025-01-15*
