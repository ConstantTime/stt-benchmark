"""STT Service configurations using factory functions.

Each service is defined as a factory function that returns a configured
Pipecat STT service instance. This gives full control over constructor
arguments for each service.

To add a new service or modify an existing one:
1. Create/modify a factory function that returns the configured service
2. Add/update the entry in STT_SERVICES with required env vars

Example - modifying Gradium to use a US endpoint:

    def create_gradium() -> FrameProcessor:
        from pipecat.services.gradium.stt import GradiumSTTService
        return GradiumSTTService(
            api_key=_get_env("GRADIUM_API_KEY"),
            api_endpoint_base_url="wss://us.api.gradium.ai/api/speech/asr",
        )
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transcriptions.language import Language

if TYPE_CHECKING:
    import aiohttp

    from stt_benchmark.models import ServiceName


def _get_env(name: str) -> str:
    """Get environment variable from config (supports .env files), raising if not set."""
    from stt_benchmark.config import get_config

    config = get_config()
    attr_name = name.lower()
    # Try to get from config first (which loads .env)
    value = getattr(config, attr_name, None)
    if value:
        return value
    # Fall back to os.getenv for env vars not in config
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"{name} environment variable not set")
    return value


# Type alias for service factory functions
ServiceFactory = Callable[..., FrameProcessor]


@dataclass
class ServiceDefinition:
    """Definition of an STT service."""

    # Factory function that creates the configured service instance
    factory: ServiceFactory

    # Environment variables required for this service
    # Used to check if service is available before attempting to create it
    required_env_vars: list[str] = field(default_factory=list)

    # Whether this service requires an aiohttp.ClientSession to be passed
    # to the factory. When True, the pipeline runner will create a session
    # context and pass it as the first argument to the factory.
    needs_aiohttp: bool = False

    # Whether this factory accepts a `language: Language` keyword argument
    # for per-sample language dispatch. Factories that hardcode a language
    # leave this False; the runner won't pass `language=` to them.
    accepts_language: bool = False


# ISO 639 / BCP-47 → pipecat Language enum.
# Covers the 6 target languages plus English. Unknown codes fall back to EN.
_LANGUAGE_MAP: dict[str, Language] = {
    # 2-letter (ISO 639-1)
    "en": Language.EN,
    "no": Language.NB,  # FLEURS uses "no" for Norwegian
    "nb": Language.NB,
    "da": Language.DA,
    "de": Language.DE,
    "fr": Language.FR,
    "it": Language.IT,
    "es": Language.ES,
    # 3-letter (ISO 639-2/3) — what smart-turn-data uses
    "eng": Language.EN,
    "nor": Language.NB,
    "nob": Language.NB,
    "dan": Language.DA,
    "deu": Language.DE,
    "ger": Language.DE,
    "fra": Language.FR,
    "fre": Language.FR,
    "ita": Language.IT,
    "spa": Language.ES,
    # BCP-47 (FLEURS subset codes)
    "nb_no": Language.NB_NO,
    "da_dk": Language.DA_DK,
    "de_de": Language.DE_DE,
    "fr_fr": Language.FR_FR,
    "it_it": Language.IT_IT,
    "es_es": Language.ES_ES,
    "es_419": Language.ES_419,
    "en_us": Language.EN_US,
}


def parse_language(code: str | None) -> Language:
    """Map a language code string to a pipecat Language enum.

    Accepts hyphens or underscores; case-insensitive. Falls back to the
    head before the separator (e.g. "nb_no" -> "nb"). Defaults to EN.
    """
    if not code:
        return Language.EN
    norm = code.lower().replace("-", "_")
    if norm in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[norm]
    head = norm.split("_", 1)[0]
    return _LANGUAGE_MAP.get(head, Language.EN)


# =============================================================================
# SERVICE FACTORY FUNCTIONS
# =============================================================================
# Each factory returns a fully configured Pipecat STT service instance.
# Modify these to change service configuration (models, endpoints, params, etc.)
# =============================================================================


def create_assemblyai() -> FrameProcessor:
    from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
    from pipecat.services.assemblyai.stt import AssemblyAISTTService

    return AssemblyAISTTService(
        api_key=_get_env("ASSEMBLYAI_API_KEY"),
        connection_params=AssemblyAIConnectionParams(
            end_of_turn_confidence_threshold=1.0,
            max_turn_silence=2000,
        ),
        vad_force_turn_endpoint=True,
    )


def create_aws() -> FrameProcessor:
    from pipecat.services.aws.stt import AWSTranscribeSTTService

    return AWSTranscribeSTTService(
        api_key=_get_env("AWS_SECRET_ACCESS_KEY"),
        aws_access_key_id=_get_env("AWS_ACCESS_KEY_ID"),
        region=_get_env("AWS_REGION"),
    )


def create_azure() -> FrameProcessor:
    from pipecat.services.azure.stt import AzureSTTService

    return AzureSTTService(
        api_key=_get_env("AZURE_SPEECH_API_KEY"),
        region=_get_env("AZURE_SPEECH_REGION"),
    )


def create_cartesia() -> FrameProcessor:
    from pipecat.services.cartesia.stt import CartesiaSTTService

    return CartesiaSTTService(
        api_key=_get_env("CARTESIA_API_KEY"),
        model="ink-whisper",
    )


def create_deepgram() -> FrameProcessor:
    from deepgram import LiveOptions
    from pipecat.services.deepgram.stt import DeepgramSTTService

    return DeepgramSTTService(
        api_key=_get_env("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-3-general",
            smart_format=False,
            profanity_filter=False,
            language=Language.EN,
        ),
    )


def create_elevenlabs(language: Language = Language.EN) -> FrameProcessor:
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService

    return ElevenLabsRealtimeSTTService(
        api_key=_get_env("ELEVENLABS_API_KEY"),
        model="scribe_v2_realtime",
        params=ElevenLabsRealtimeSTTService.InputParams(
            language=language,
        ),
    )


def create_elevenlabs_http(
    aiohttp_session: "aiohttp.ClientSession", language: Language = Language.EN
) -> FrameProcessor:
    from pipecat.services.elevenlabs.stt import ElevenLabsSTTService

    return ElevenLabsSTTService(
        aiohttp_session=aiohttp_session,
        api_key=_get_env("ELEVENLABS_API_KEY"),
        model="scribe_v2",
        params=ElevenLabsSTTService.InputParams(
            language=language,
        ),
    )


def create_fal() -> FrameProcessor:
    from pipecat.services.fal.stt import FalSTTService

    return FalSTTService(
        api_key=_get_env("FAL_KEY"),
        params=FalSTTService.InputParams(
            language=Language.EN,
        ),
    )


def create_gladia() -> FrameProcessor:
    from pipecat.services.gladia.config import (
        GladiaInputParams,
        LanguageConfig,
        PreProcessingConfig,
    )
    from pipecat.services.gladia.stt import GladiaSTTService

    return GladiaSTTService(
        api_key=_get_env("GLADIA_API_KEY"),
        region=os.getenv("GLADIA_REGION", "us-west"),
        model="solaria-1",
        params=GladiaInputParams(
            language_config=LanguageConfig(
                languages=[Language.EN],
            ),
            endpointing=0.01,
            pre_processing=PreProcessingConfig(
                speech_threshold=0.8,
            ),
        ),
    )


def create_google() -> FrameProcessor:
    from pipecat.services.google.stt import GoogleSTTService

    return GoogleSTTService(
        credentials_path=_get_env("GOOGLE_APPLICATION_CREDENTIALS"),
        location=os.getenv("GOOGLE_LOCATION", "us-central1"),
        params=GoogleSTTService.InputParams(
            languages=Language.EN_US,
            model="latest_long",
        ),
    )


def create_gradium() -> FrameProcessor:
    from pipecat.services.gradium.stt import GradiumSTTService

    return GradiumSTTService(
        api_key=_get_env("GRADIUM_API_KEY"),
        api_endpoint_base_url=os.getenv(
            "GRADIUM_BASE_URL", "wss://us.api.gradium.ai/api/speech/asr"
        ),
        params=GradiumSTTService.InputParams(
            language=Language.EN,
        ),
    )


def create_groq() -> FrameProcessor:
    from pipecat.services.groq.stt import GroqSTTService

    return GroqSTTService(
        api_key=_get_env("GROQ_API_KEY"),
        model="whisper-large-v3-turbo",
        language=Language.EN,
    )


def create_hathora() -> FrameProcessor:
    from pipecat.services.hathora.stt import HathoraSTTService

    return HathoraSTTService(
        api_key=_get_env("HATHORA_API_KEY"),
        model="nvidia-parakeet-tdt-0.6b-v3",
    )


def create_nvidia() -> FrameProcessor:
    from pipecat.services.nvidia.stt import NvidiaSTTService

    return NvidiaSTTService(
        api_key=_get_env("NVIDIA_API_KEY"),
        params=NvidiaSTTService.InputParams(
            language=Language.EN_US,
        ),
    )


def create_openai() -> FrameProcessor:
    from pipecat.services.openai.stt import OpenAISTTService

    return OpenAISTTService(
        api_key=_get_env("OPENAI_API_KEY"),
        model="gpt-4o-mini-transcribe",
        language=Language.EN,
    )


def create_openai_realtime() -> FrameProcessor:
    from pipecat.services.openai.stt import OpenAIRealtimeSTTService

    return OpenAIRealtimeSTTService(
        api_key=_get_env("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
        language=Language.EN,
    )


def create_sambanova() -> FrameProcessor:
    from pipecat.services.sambanova.stt import SambaNovaSTTService

    return SambaNovaSTTService(
        api_key=_get_env("SAMBANOVA_API_KEY"),
        model="Whisper-Large-v3",
        language=Language.EN,
    )


def create_sarvam() -> FrameProcessor:
    from pipecat.services.sarvam.stt import SarvamSTTService

    return SarvamSTTService(
        api_key=_get_env("SARVAM_API_KEY"),
        model="saarika:v2.5",
    )


def create_soniox() -> FrameProcessor:
    from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService

    return SonioxSTTService(
        api_key=_get_env("SONIOX_API_KEY"),
        params=SonioxInputParams(
            model="stt-rt-v4",
            language_hints=[Language.EN],
            language_hints_strict=True,
        ),
        vad_force_turn_endpoint=True,
    )


def create_speechmatics(language: Language = Language.EN) -> FrameProcessor:
    from pipecat.services.speechmatics.stt import SpeechmaticsSTTService, TurnDetectionMode

    return SpeechmaticsSTTService(
        api_key=_get_env("SPEECHMATICS_API_KEY"),
        base_url=os.getenv("SPEECHMATICS_RT_URL", "wss://us.rt.speechmatics.com/v2"),
        params=SpeechmaticsSTTService.InputParams(
            language=language,
            turn_detection_mode=TurnDetectionMode.EXTERNAL,
        ),
    )


def create_whisper() -> FrameProcessor:
    from pipecat.services.whisper.stt import Model, WhisperSTTService

    return WhisperSTTService(
        model=Model.DISTIL_MEDIUM_EN,
        language=Language.EN,
    )


def create_xai(
    aiohttp_session: "aiohttp.ClientSession", language: Language = Language.EN
) -> FrameProcessor:
    from stt_benchmark.xai_stt import XAIRealtimeSTTService

    return XAIRealtimeSTTService(
        aiohttp_session=aiohttp_session,
        api_key=_get_env("XAI_API_KEY"),
        base_url=os.getenv("XAI_STT_BASE_URL", "wss://api.x.ai/v1/stt"),
        language=language,
        endpointing_ms=int(os.getenv("XAI_ENDPOINTING_MS", "10")),
        interim_results=True,
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================
# Maps service names to their definitions (factory + required env vars).
# The required_env_vars are used to check availability before creating.
# =============================================================================

STT_SERVICES: dict[str, ServiceDefinition] = {
    "assemblyai": ServiceDefinition(
        factory=create_assemblyai,
        required_env_vars=["ASSEMBLYAI_API_KEY"],
    ),
    "aws": ServiceDefinition(
        factory=create_aws,
        required_env_vars=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    ),
    "azure": ServiceDefinition(
        factory=create_azure,
        required_env_vars=["AZURE_SPEECH_API_KEY", "AZURE_SPEECH_REGION"],
    ),
    "cartesia": ServiceDefinition(
        factory=create_cartesia,
        required_env_vars=["CARTESIA_API_KEY"],
    ),
    "deepgram": ServiceDefinition(
        factory=create_deepgram,
        required_env_vars=["DEEPGRAM_API_KEY"],
    ),
    "elevenlabs": ServiceDefinition(
        factory=create_elevenlabs,
        required_env_vars=["ELEVENLABS_API_KEY"],
        accepts_language=True,
    ),
    "elevenlabs_http": ServiceDefinition(
        factory=create_elevenlabs_http,
        required_env_vars=["ELEVENLABS_API_KEY"],
        needs_aiohttp=True,
        accepts_language=True,
    ),
    "fal": ServiceDefinition(
        factory=create_fal,
        required_env_vars=["FAL_KEY"],
    ),
    "gladia": ServiceDefinition(
        factory=create_gladia,
        required_env_vars=["GLADIA_API_KEY"],
    ),
    "google": ServiceDefinition(
        factory=create_google,
        required_env_vars=["GOOGLE_APPLICATION_CREDENTIALS"],
    ),
    "gradium": ServiceDefinition(
        factory=create_gradium,
        required_env_vars=["GRADIUM_API_KEY"],
    ),
    "groq": ServiceDefinition(
        factory=create_groq,
        required_env_vars=["GROQ_API_KEY"],
    ),
    "hathora": ServiceDefinition(
        factory=create_hathora,
        required_env_vars=["HATHORA_API_KEY"],
    ),
    "nvidia": ServiceDefinition(
        factory=create_nvidia,
        required_env_vars=["NVIDIA_API_KEY"],
    ),
    "openai": ServiceDefinition(
        factory=create_openai,
        required_env_vars=["OPENAI_API_KEY"],
    ),
    "openai_realtime": ServiceDefinition(
        factory=create_openai_realtime,
        required_env_vars=["OPENAI_API_KEY"],
    ),
    "sambanova": ServiceDefinition(
        factory=create_sambanova,
        required_env_vars=["SAMBANOVA_API_KEY"],
    ),
    "sarvam": ServiceDefinition(
        factory=create_sarvam,
        required_env_vars=["SARVAM_API_KEY"],
    ),
    "soniox": ServiceDefinition(
        factory=create_soniox,
        required_env_vars=["SONIOX_API_KEY"],
    ),
    "speechmatics": ServiceDefinition(
        factory=create_speechmatics,
        required_env_vars=["SPEECHMATICS_API_KEY"],
        accepts_language=True,
    ),
    "whisper": ServiceDefinition(
        factory=create_whisper,
        required_env_vars=[],  # Local model, no API key needed
    ),
    "xai": ServiceDefinition(
        factory=create_xai,
        required_env_vars=["XAI_API_KEY"],
        needs_aiohttp=True,
        accepts_language=True,
    ),
}

SERVICE_ALIASES = {
    "grok": "xai",
}


# =============================================================================
# SERVICE CREATION & AVAILABILITY
# =============================================================================


def get_service_definition(name: str) -> ServiceDefinition:
    """Get the service definition by name."""
    if name not in STT_SERVICES:
        raise ValueError(f"Unknown service: {name}. Available: {list(STT_SERVICES.keys())}")
    return STT_SERVICES[name]


def get_all_service_names() -> list[str]:
    """Get all configured service names."""
    return list(STT_SERVICES.keys())


def _get_env_from_config(env_var_name: str) -> str:
    """Get environment variable value from config (supports .env files via Pydantic).

    Derives config attribute from env var name: DEEPGRAM_API_KEY -> deepgram_api_key
    Falls back to os.getenv() for env vars not in config.
    """
    from stt_benchmark.config import get_config

    config = get_config()
    attr_name = env_var_name.lower()
    # Try to get from config first (which loads .env)
    value = getattr(config, attr_name, None)
    if value is not None:
        return value
    # Fall back to os.getenv for env vars not in config
    return os.getenv(env_var_name, "")


def is_service_available(name: str) -> bool:
    """Check if a service has all required environment variables set."""
    if name not in STT_SERVICES:
        return False
    definition = STT_SERVICES[name]
    return all(_get_env_from_config(env_var) for env_var in definition.required_env_vars)


def create_stt_service(
    service_name: "ServiceName",
    aiohttp_session: "aiohttp.ClientSession | None" = None,
    language: Language | None = None,
) -> FrameProcessor:
    """Create an STT service instance using its factory function.

    Args:
        service_name: The STT service to create.
        aiohttp_session: Optional aiohttp session for services that require one
            (i.e. services with needs_aiohttp=True in their ServiceDefinition).
        language: Optional pipecat Language to dispatch this transcription in.
            Only passed to factories with `accepts_language=True`; others
            fall back to whatever language they hardcode.

    Returns:
        Configured STT service instance.

    Raises:
        ValueError: If service_name is not supported or required credentials are missing.
    """
    from loguru import logger

    definition = get_service_definition(service_name.value)
    logger.debug(
        f"Creating {service_name.value} STT service"
        + (f" (language={language})" if language and definition.accepts_language else "")
    )

    kwargs = {}
    if definition.accepts_language and language is not None:
        kwargs["language"] = language

    if definition.needs_aiohttp:
        if aiohttp_session is None:
            raise ValueError(
                f"Service {service_name.value} requires an aiohttp session "
                f"but none was provided. The pipeline runner should create one."
            )
        return definition.factory(aiohttp_session, **kwargs)

    return definition.factory(**kwargs)


def get_available_services() -> list["ServiceName"]:
    """Get list of services that have all required credentials configured.

    Returns:
        List of ServiceName values for available services.
    """
    from loguru import logger

    from stt_benchmark.models import ServiceName

    available = []
    for name in STT_SERVICES:
        if is_service_available(name):
            try:
                available.append(ServiceName(name))
            except ValueError:
                logger.warning(f"Service {name} not in ServiceName enum")
        else:
            definition = STT_SERVICES[name]
            logger.debug(
                f"Service {name} not available (missing env vars: {definition.required_env_vars})"
            )
    return available


def get_all_services() -> list["ServiceName"]:
    """Get list of all supported services.

    Returns:
        List of all ServiceName values.
    """
    from stt_benchmark.models import ServiceName

    return list(ServiceName)


# =============================================================================
# CLI UTILITIES
# =============================================================================


def parse_service_name(name: str) -> "ServiceName":
    """Parse a service name string to ServiceName enum.

    Accepts any name in STT_SERVICES (Pipecat-pipeline services) as well as
    any value in the ServiceName enum (e.g. batch-REST services like
    `speechmatics_batch` that don't have a Pipecat factory but do produce
    rows in the `results` table for `wer` and `report` to consume).

    Args:
        name: Service name (case-insensitive)

    Returns:
        ServiceName enum value

    Raises:
        ValueError: If the name is not a recognized service.
    """
    from stt_benchmark.models import ServiceName

    name_lower = name.strip().lower()
    resolved_name = SERVICE_ALIASES.get(name_lower, name_lower)
    if resolved_name in STT_SERVICES:
        return ServiceName(resolved_name)
    # Allow ServiceName members that aren't in STT_SERVICES (batch services).
    try:
        return ServiceName(resolved_name)
    except ValueError:
        valid = sorted(set(STT_SERVICES.keys()) | {s.value for s in ServiceName})
        raise ValueError(
            f"Unknown service: {name}. Available: {', '.join(valid)}"
        ) from None


def parse_services_arg(services_arg: str) -> list["ServiceName"]:
    """Parse a comma-separated services argument.

    Args:
        services_arg: Comma-separated service names or 'all'

    Returns:
        List of ServiceName enum values
    """
    if services_arg.lower() == "all":
        return get_available_services()

    return [parse_service_name(s) for s in services_arg.split(",")]
