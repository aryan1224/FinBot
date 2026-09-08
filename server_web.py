import asyncio
import random
import json
import os
import time
from dataclasses import dataclass

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from pipecat.adapters.schemas.tools_schema import FunctionSchema
from pipecat.audio.vad.vad_analyzer import VADParams 
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

# The output transport does NOT auto-serialize arbitrary custom Frame
# subclasses -- only audio frames and this specific message-frame family
# get pushed through send_message() -> serializer.serialize() -> the
# websocket. This is pipecat's actual documented mechanism for sending
# arbitrary out-of-band data (JSON, in our case) to the client. The class
# was renamed TransportMessageFrame -> OutputTransportMessageFrame /
# TransportMessageUrgentFrame -> OutputTransportMessageUrgentFrame in
# newer pipecat releases, so we try both names.
try:
    from pipecat.frames.frames import (
        OutputTransportMessageFrame,
        OutputTransportMessageUrgentFrame,
    )
except ImportError:  # older pipecat releases
    from pipecat.frames.frames import TransportMessageFrame as OutputTransportMessageFrame
    from pipecat.frames.frames import (
        TransportMessageUrgentFrame as OutputTransportMessageUrgentFrame,
    )
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.llm_service import FunctionCallResultProperties

# FrameSerializerType was dropped from base_serializer in newer pipecat
# releases (e.g. 1.7.0) — only import it if it's actually there.
try:
    from pipecat.serializers.base_serializer import FrameSerializerType
except ImportError:
    FrameSerializerType = None
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService

# FastAPIWebsocketTransport moved modules across pipecat versions — try both.
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from modules.embedding import preload_embedding_model
from modules.ingestion import ingest_pdf
from modules.retrieval import retrieve_top_k
from contextlib import asynccontextmanager
from modules.database import get_products

load_dotenv()

SARVAM_API_KEY = os.getenv("API_KEY")
if not SARVAM_API_KEY:
    raise RuntimeError("API_KEY is missing from .env")

AUDIO_IN_SAMPLE_RATE = 16000
AUDIO_OUT_SAMPLE_RATE = 24000

import json as _json


class _ResilientJSON:
    """Sarvam's streaming completions occasionally resend the full
    arguments string more than once for near-empty tool calls (e.g. '{}{}'
    instead of '{}'), which breaks pipecat's naive json.loads(arguments) in
    BaseOpenAILLMService._process_context. We recover by decoding only the
    first complete JSON value and discarding anything appended after it."""

    JSONDecodeError = _json.JSONDecodeError

    @staticmethod
    def loads(s, *args, **kwargs):
        try:
            return _json.loads(s, *args, **kwargs)
        except _json.JSONDecodeError:
            obj, _ = _json.JSONDecoder().raw_decode(s)
            return obj


import pipecat.services.openai.base_llm as _base_llm

_base_llm.json = _ResilientJSON()


# ================================================================
# LLM CONNECTION WARM-UP
# ================================================================


async def warm_up_llm():
    """Fire a throwaway 1-token completion at Sarvam as soon as the client
    connects, before the user has said anything. Profiling showed turn 1's
    LLM TTFB running well above later turns' despite similar or smaller
    completions -- consistent with Sarvam-side model routing/warm-up cost
    on a cold connection rather than our pipeline config. This absorbs that
    cost before the real turn 1 request, on a separate connection so it
    can't interfere with the pipeline's own LLM calls. Fire-and-forget:
    failures here should never block or crash the real conversation."""
    t0 = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.sarvam.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {SARVAM_API_KEY}"},
                json={
                    "model": "sarvam-105b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                await resp.read()
        logger.info(f"[latency] LLM warm-up completed in {(time.monotonic() - t0) * 1000:.0f} ms")
    except Exception as e:
        logger.warning(f"[latency] LLM warm-up failed (non-fatal): {e}")


# ================================================================
# LATENCY INSTRUMENTATION
# ================================================================


@dataclass
class LatencyReportFrame(Frame):
    """Kept only as an internal marker type; no longer pushed directly
    down the pipeline (see LatencyStepProbe, which now wraps this data in
    an OutputTransportMessageUrgentFrame -- the only frame family the
    output transport actually forwards to the serializer for non-audio
    data)."""

    step: str
    label: str
    ms: float


# ================================================================
# LIVE TRANSCRIPT / BOT-TEXT TAPS
# ================================================================
#
# context_aggregator.user() consumes TranscriptionFrame/
# InterimTranscriptionFrame to build conversation history and does not
# forward them further down the pipeline; context_aggregator.assistant()
# does the same with TTSTextFrame. So this text needs to be captured
# *before* each aggregator. It's then sent to the browser wrapped in
# OutputTransportMessageUrgentFrame(message={...}) -- a plain custom Frame
# subclass here would NOT reach the serializer; only audio frames and this
# specific message-frame family get forwarded by the output transport's
# send_message() path.


class TranscriptTap(FrameProcessor):
    """Placed before context_aggregator.user(). Sends interim/final STT
    text to the browser as a JSON transport message, then forwards the
    original frame unchanged so normal context-building still happens."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterimTranscriptionFrame):
            logger.info(f"[transcript-tap] interim: {frame.text!r}")
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"type": "interim_transcript", "text": frame.text}
                ),
                direction,
            )
        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"[transcript-tap] final: {frame.text!r}")
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"type": "final_transcript", "text": frame.text}
                ),
                direction,
            )
        await self.push_frame(frame, direction)


class BotTextTap(FrameProcessor):
    """Placed before context_aggregator.assistant(). Same idea as
    TranscriptTap, for the streamed TTS reply text -- with one important
    difference: this uses OutputTransportMessageFrame (non-urgent), NOT
    the Urgent variant. Urgent messages jump the output transport's queue
    and get sent immediately, ahead of whatever audio is still being
    paced out -- since TTS generates all of a reply's text well before
    its audio has finished streaming/playing, that made bot text arrive
    in the browser in one burst, well ahead of the audio it's supposed to
    accompany. Non-urgent messages travel through the same paced queue as
    the audio frames around them, so each chunk lands roughly when its
    matching audio does."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame):
            logger.info(f"[bot-text-tap] chunk: {frame.text!r}")
            await self.push_frame(
                OutputTransportMessageFrame(message={"type": "bot_text", "text": frame.text}),
                direction,
            )
        await self.push_frame(frame, direction)


# ================================================================
# LATENCY INSTRUMENTATION (probes)
# ================================================================
#
# One `latency_state` dict is created per WebSocket connection (see
# run_bot) and shared by every probe below. It is cleared at the start of
# each user turn — anchored on the final STT transcript, see
# TurnStartProbe — and keys are timestamps (time.monotonic(), in seconds)
# recorded the first time a given frame type is seen during that turn.


class VadEventLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("[latency] VAD started")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("[latency] VAD stopped")

        await self.push_frame(frame, direction)


class TurnStartProbe(FrameProcessor):
    def __init__(self, state: dict):
        super().__init__()
        self._state = state

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            now = time.monotonic()

            # Start a fresh latency measurement for this user turn.
            self._state.clear()
            self._state["transcript_ready"] = now

            logger.info("[latency] final STT transcript ready")

        await self.push_frame(frame, direction)


class LatencyStepProbe(FrameProcessor):
    """Fires once per turn, the first time a frame of `frame_types` passes
    through. Records time.monotonic() under `key` in the shared `state`
    dict, and if `since_key` is already present in `state`, logs (and
    optionally emits to the browser) the elapsed time between the two.

    Still used for bot_started_probe below (transcript_ready -> bot
    actually starts speaking), which is unaffected by the tool-call /
    fast-path attribution issue LLMTokenProbe and TTSFirstAudioProbe
    exist to fix."""

    def __init__(
        self,
        state: dict,
        key: str,
        frame_types,
        since_key: str | None = None,
        label: str | None = None,
        emit: bool = False,
    ):
        super().__init__()
        self._state = state
        self._key = key
        self._frame_types = frame_types
        self._since_key = since_key
        self._label = label or key
        self._emit = emit

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, self._frame_types) and self._key not in self._state:
            now = time.monotonic()
            self._state[self._key] = now

            if self._since_key and self._since_key in self._state:
                ms = (now - self._state[self._since_key]) * 1000
                logger.info(f"[latency] {self._label}: {ms:.0f} ms")
                if self._emit:
                    await self.push_frame(
                        OutputTransportMessageUrgentFrame(
                            message={
                                "type": "latency",
                                "step": self._key,
                                "label": self._label,
                                "ms": round(ms, 1),
                            }
                        ),
                        direction,
                    )
            else:
                logger.info(f"[latency] {self._label} reached")

        await self.push_frame(frame, direction)


class LLMTokenProbe(FrameProcessor):
    """Tracks LLM output across up to two sequential completions per turn.
    Tool-call turns make two LLM calls: one that decides to call a tool
    (llm1_first_token), and — for tools NOT on the templated fast-path —
    a second one that generates the actual spoken reply after the tool
    result comes back (llm2_first_token). Non-tool turns, and fast-pathed
    tool turns (get_balance / get_payment_status / get_payment_policy, all
    of which skip the second LLM call via
    FunctionCallResultProperties(run_llm=False)), only ever produce
    llm1_first_token — which in the non-tool case IS the reply, and in the
    fast-path case is the tool-call decision, with the actual spoken text
    coming from a manually-pushed TTSSpeakFrame rather than a second
    completion.

    This replaces an earlier single-probe design, which always anchored
    TTS timing on the FIRST LLM token — silently absorbing the entire
    second LLM call (~370-400ms) into what got logged as "TTS latency" on
    turns that weren't fast-pathed."""

    def __init__(self, state: dict):
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.monotonic()

        if isinstance(frame, FunctionCallResultFrame) and "tool_result" not in self._state:
            self._state["tool_result"] = now
            if "llm1_first_token" in self._state:
                ms = (now - self._state["llm1_first_token"]) * 1000
                logger.info(f"[latency] tool round-trip (decision -> result back): {ms:.0f} ms")

        elif isinstance(frame, TextFrame):
            if "llm1_first_token" not in self._state:
                self._state["llm1_first_token"] = now
                if "transcript_ready" in self._state:
                    ms = (now - self._state["transcript_ready"]) * 1000
                    logger.info(
                        f"[latency] LLM #1 (transcript ready -> first token / tool-call decision): {ms:.0f} ms"
                    )
            elif "tool_result" in self._state and "llm2_first_token" not in self._state:
                self._state["llm2_first_token"] = now
                ms = (now - self._state["tool_result"]) * 1000
                logger.info(f"[latency] LLM #2 (tool result -> first token of actual reply): {ms:.0f} ms")

        await self.push_frame(frame, direction)


class TTSFirstAudioProbe(FrameProcessor):
    """Measures TTS latency against the correct reply-start anchor:
    llm2_first_token when a second LLM call happened, otherwise
    llm1_first_token (every non-tool turn, and every fast-pathed tool turn
    where the spoken text comes from a manually pushed TTSSpeakFrame
    rather than a second completion). Fixes the earlier artifact where TTS
    silently absorbed a full second LLM call on tool turns."""

    def __init__(self, state: dict):
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and "tts_first_audio" not in self._state:
            now = time.monotonic()
            self._state["tts_first_audio"] = now
            reply_key = "llm2_first_token" if "llm2_first_token" in self._state else "llm1_first_token"
            if reply_key in self._state:
                ms = (now - self._state[reply_key]) * 1000
                logger.info(f"[latency] TTS (reply text ready -> first audio byte): {ms:.0f} ms")
        await self.push_frame(frame, direction)


class MetricsLogger(FrameProcessor):
    """Logs pipecat's own built-in per-service TTFB / usage metrics, which
    are emitted automatically because PipelineParams sets
    enable_metrics=True and enable_usage_metrics=True below.

    Also flags a specific anomaly caught in earlier testing: a turn where
    the LLM burned a large number of completion tokens (~380-549) to
    produce a short spoken reply (~20 characters) -- caused by
    sarvam-105b's reasoning mode being on by default, now disabled via
    extra={"reasoning_effort": None} on OpenAILLMService.Settings below.
    Confirmed fixed as of this session (completion tokens now land 9-18
    per turn), but left in place as a regression guard."""

    TOKEN_BLOAT_THRESHOLD = 150  # generous for a short, spoken voice reply

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                logger.info(f"[metrics] {d}")
                if (
                    isinstance(d, LLMUsageMetricsData)
                    and d.value.completion_tokens > self.TOKEN_BLOAT_THRESHOLD
                ):
                    logger.warning(
                        f"[latency] possible token bloat: {d.value.completion_tokens} "
                        f"completion tokens for what should be a short spoken reply "
                        f"(model={d.model}). This is likely where an outsized 'LLM' "
                        f"latency number is coming from, not slow generation per se."
                    )
        await self.push_frame(frame, direction)


# ================================================================
# CUSTOM SERIALIZER — raw PCM binary audio in both directions, plus
# JSON text events for live transcript / call-state / latency.
# ================================================================


class BrowserFrameSerializer(FrameSerializer):
    """Bridges pipecat frames <-> a plain browser WebSocket.

    Mic audio arrives as raw binary PCM16 chunks (no header, no JSON
    wrapper) and is turned straight into InputAudioRawFrames. Outgoing
    TTS audio is sent the same way. Everything else useful for a live
    transcript UI (interim/final STT text, streamed bot text, speaking
    state, latency measurements) is serialized as a small JSON string.

    Live transcript/bot-text/latency events are carried by
    OutputTransportMessageUrgentFrame(message={...}) -- NOT by the raw
    InterimTranscriptionFrame / TranscriptionFrame / TTSTextFrame types.
    Those get consumed by the context aggregators before they ever reach
    this serializer, and even a plain custom Frame subclass wouldn't be
    forwarded by the output transport for non-audio data -- only audio
    frames and this specific message-frame family are.
    """

    def __init__(
        self,
        audio_in_sample_rate: int = AUDIO_IN_SAMPLE_RATE,
        audio_out_sample_rate: int = AUDIO_OUT_SAMPLE_RATE,
    ):
        self._audio_in_sample_rate = audio_in_sample_rate
        self._audio_out_sample_rate = audio_out_sample_rate

    @property
    def type(self):
        # Older pipecat versions require this to return a FrameSerializerType
        # enum member; newer ones (>=1.x) dropped the enum. Handle both.
        if FrameSerializerType is not None:
            return FrameSerializerType.BINARY
        return "binary"

    async def setup(self, frame: StartFrame):
        pass

    async def serialize(self, frame: Frame):
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            # frame.message is the plain dict we built in the taps/probes
            # above -- e.g. {"type": "final_transcript", "text": "..."}.
            logger.info(f"[ws-out] {frame.message}")
            return json.dumps(frame.message)

        if isinstance(frame, UserStartedSpeakingFrame):
            return json.dumps({"type": "user_started_speaking"})

        if isinstance(frame, UserStoppedSpeakingFrame):
            return json.dumps({"type": "user_stopped_speaking"})

        if isinstance(frame, BotStartedSpeakingFrame):
            return json.dumps({"type": "bot_started_speaking"})

        if isinstance(frame, BotStoppedSpeakingFrame):
            return json.dumps({"type": "bot_stopped_speaking"})

        return None

    async def deserialize(self, data):
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=self._audio_in_sample_rate,
                num_channels=1,
            )
        # We don't currently expect text control messages from the
        # client — mic on/off is handled purely client-side by the
        # tap-to-talk button (it just stops sending audio chunks).
        return None


# ================================================================
# BANK DATA + TOOLS + SYSTEM PROMPT
# ================================================================

PAYMENT_DB = {
    "aryan": {"amount": 500, "status": "successful"},
    "ankit": {"amount": 1200, "status": "failed", "reason": "insufficient balance"},
    "amit": {"amount": 300, "status": "successful"},
    "vivek": {"amount": 750, "status": "failed", "reason": "online payment disabled"},
    "yash": {"amount": 2000, "status": "successful"},
}

BALANCE_DB = {
    "aryan": 15400,
    "ankit": 2300,
    "amit": 8900,
    "vivek": 500,
    "yash": 42000,
}

# Spoken fallback used only when the policy vector DB doesn't exist yet
# (i.e. no policy PDF has ever been uploaded via /upload) or when the RAG
# LLM call itself fails. Language-aware so Hindi users get a Hindi fallback.
_POLICY_UNAVAILABLE = {
    "en": (
        "I don't have policy information available right now. "
        "Please contact support at 1800-200-1234."
    ),
    "hi": (
        "अभी पॉलिसी जानकारी उपलब्ध नहीं है। "
        "कृपया 1800-200-1234 पर संपर्क करें।"
    ),
}


def _policy_unavailable_msg(lang: str) -> str:
    return _POLICY_UNAVAILABLE.get(lang, _POLICY_UNAVAILABLE["en"])

SYSTEM_PROMPT = """
You are FinBot, a warm and friendly voice assistant for IndiaFirst Bank.

At the beginning of the conversation, greet the user and ask for their name.
After they provide their name, say "Hi <name>, how can I help you today?"

You can help with:
1. Payment or transaction status.
2. Account balance.
3. IndiaFirst Bank payment policies.
4. Financial product recommendations.

If the user asks about payment status, call the get_payment_status tool.
If the user asks about account balance, call the get_balance tool.
If the user asks about payment policy, retry limits, failed-payment rules, or
any other policy question, call the get_payment_policy tool and pass the
user's question as the "query" argument, in their own words.

If the user wants to find, choose, compare, or get a recommendation for a
financial product, call the get_product_recommendations tool.

Use only product filters that are clearly implied by the user's request.
Do not invent filter values.

Valid product domains are:
Banking, Lending, Investments, Insurance, Cards, Retirement.

If the user gives only a broad requirement, use only the filter that is known.
For example, if the user says they want an investment product, call
get_product_recommendations with product_domain="Investments" without
inventing a product type.

After the product tool returns products, recommend the product that best
matches the user's needs using only the returned product information.

Do not invent or guess product features, pricing, eligibility, benefits,
or other product details.

If multiple products are suitable, briefly explain the difference and
recommend the closest fit.

If no products are returned, say you could not find a matching product and
ask the user to refine their requirements.

Do not say anything before calling a tool. Do not say things like "please wait"
or "let me check" — just call the tool directly and silently. Only speak once
you have the tool result, then answer naturally and briefly using that real
data. Never write out, describe, or type a tool call, function name, or JSON
as part of your spoken reply — your reply text should only ever contain
natural spoken language, and only after the tool result is available.

Do not invent or guess payment, balance, policy, or product information.

If a payment failed, explain the reason and suggest the customer contact support
at 1800-200-1234.

Respond in the same language as the user. Hindi, English, and Hinglish are all
supported.

Keep responses short and conversational — prefer a single short sentence,
and only use a second sentence if it's truly necessary. Never use markdown,
bullet points, emojis, or any symbols as your replies are converted directly
to speech by a text-to-speech engine that can only pronounce plain words in
Hindi or English.
"""

async def get_payment_status(params):
    """Templated fast-path: composes the spoken reply directly from the
    tool result and pushes it straight to TTS via TTSSpeakFrame, skipping
    the second LLM call that would otherwise exist purely to reformat this
    same data into a sentence (~300-400ms per turn measured this session).
    run_llm=False tells pipecat's context aggregator not to trigger that
    second completion. TTSSpeakFrame's append_to_context=True default
    keeps reply_text in conversation history, so a follow-up like "what
    was that payment status again?" still has it available."""
    t0 = time.monotonic()
    name = params.arguments.get("name", "").strip().lower()
    logger.info(f"Payment status lookup: {name}")

    record = PAYMENT_DB.get(name)
    if record is None:
        result = {"found": False, "message": f"No payment record found for {name}."}
        reply_text = f"I couldn't find a payment record for {name}."
    elif record["status"] == "successful":
        result = {"found": True, "customer": name, **record}
        reply_text = f"Your payment of {record['amount']} rupees was successful."
    else:
        result = {"found": True, "customer": name, **record}
        reply_text = (
            f"Your payment of {record['amount']} rupees failed due to {record['reason']}. "
            "Please contact support at 1800-200-1234."
        )

    logger.info(f"[latency] tool get_payment_status: {(time.monotonic() - t0) * 1000:.0f} ms")

    await params.result_callback(
        json.dumps(result),
        properties=FunctionCallResultProperties(run_llm=False),
    )
    await params.llm.push_frame(TTSSpeakFrame(text=reply_text))

# Filler sentences for slow tool backends, keyed by category and language.
# The actual filler is chosen at runtime by the per-connection closures in
# run_bot(), which close over a lang_state dict updated by LanguageTracker.
_FILLERS = {
    "product": {
        "en": [
            "Let me look that up for you.",
            "Sure, give me just a moment.",
            "I'll find the right product for you.",
        ],
        "hi": [
            "एक पल, मैं देखता हूँ।",
            "ज़रूर, मैं अभी चेक करता हूँ।",
            "एक सेकंड, मैं आपके लिए देखता हूँ।",
        ],
    },
    "policy": {
        "en": [
            "One moment while I check our policy documents.",
            "Let me pull up the policy for you.",
            "I'll check that in our records right away.",
        ],
        "hi": [
            "एक पल, मैं हमारी पॉलिसी देखता हूँ।",
            "ज़रूर, मैं नियम चेक करता हूँ।",
            "एक सेकंड, मैं पॉलिसी देखता हूँ।",
        ],
    },
}


def _pick_filler(category: str, lang: str) -> str:
    """Return a random filler string for the given tool category and language."""
    lang_key = "hi" if lang == "hi" else "en"
    return random.choice(_FILLERS[category][lang_key])


def _detect_lang(text: str) -> str:
    """Return 'hi' if the text contains Devanagari characters, else 'en'."""
    return "hi" if any("\u0900" <= ch <= "\u097F" for ch in text) else "en"


class LanguageTracker(FrameProcessor):
    """Updates lang_state['lang'] to 'hi' or 'en' based on the script of
    each final STT transcript. Placed just after TranscriptTap so it sees
    every finalised user utterance before the LLM call starts."""

    def __init__(self, lang_state: dict):
        super().__init__()
        self._state = lang_state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            detected = _detect_lang(frame.text)
            if detected != self._state.get("lang"):
                logger.info(f"[lang] detected language change -> {detected!r}")
            self._state["lang"] = detected
        await self.push_frame(frame, direction)


async def get_product_recommendations(params, lang_state: dict):
    """DB query + second LLM call. Speaks a random filler in the user's
    detected language while the backend work runs."""
    filler = _pick_filler("product", lang_state.get("lang", "en"))
    await params.llm.push_frame(TTSSpeakFrame(text=filler, append_to_context=False))

    filters = params.arguments

    products = await asyncio.to_thread(
        get_products,
        product_domain=filters.get("product_domain"),
        product_type=filters.get("product_type"),
        target_customer=filters.get("target_customer"),
        region=filters.get("region"),
    )

    if not products:
        result = {
            "found": False,
            "products": [],
            "reply_language": lang_state.get("lang", "en"),
        }
    else:
        result = {
            "found": True,
            "products": products,
            # Explicit cue for the pipeline LLM: product data is in English
            # but the reply must be in the user's detected language.
            "reply_language": lang_state.get("lang", "en"),
        }

    await params.result_callback(
        json.dumps(result),
        properties=FunctionCallResultProperties(run_llm=True),
    )


async def get_balance(params):
    """Templated fast-path — see get_payment_status docstring above for
    the mechanism; same pattern applied here."""
    t0 = time.monotonic()
    name = params.arguments.get("name", "").strip().lower()
    logger.info(f"Balance lookup: {name}")

    balance = BALANCE_DB.get(name)
    if balance is None:
        result = {"found": False, "message": f"No account found for {name}."}
        reply_text = f"I couldn't find an account for {name}."
    else:
        result = {"found": True, "customer": name, "balance": balance}
        reply_text = f"Your current account balance is {balance} rupees."

    logger.info(f"[latency] tool get_balance: {(time.monotonic() - t0) * 1000:.0f} ms")

    await params.result_callback(
        json.dumps(result),
        properties=FunctionCallResultProperties(run_llm=False),
    )
    await params.llm.push_frame(TTSSpeakFrame(text=reply_text))


async def call_rag_llm(prompt: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.sarvam.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {SARVAM_API_KEY}"},
                json={
                    "model": "sarvam-105b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "reasoning_effort": None,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"[rag] LLM call failed: {e}")
        return (
            "I don't have policy information available right now. "
            "Please contact support at 1800-200-1234."
        )


async def get_payment_policy(params, lang_state: dict):
    """RAG retrieval + secondary LLM call. Speaks a random filler in the
    user's detected language while the backend work runs."""
    t0 = time.monotonic()
    query = params.arguments.get("query", "").strip()
    lang = lang_state.get("lang", "en")  # resolved once, used throughout
    logger.info(f"Payment policy RAG lookup: {query!r} (lang={lang!r})")

    filler = _pick_filler("policy", lang)
    await params.llm.push_frame(TTSSpeakFrame(text=filler, append_to_context=False))

    try:
        chunks = await asyncio.to_thread(retrieve_top_k, query, 5)
    except RuntimeError as e:
        logger.warning(f"[rag] {e}")
        reply_text = _policy_unavailable_msg(lang)
        await params.result_callback(
            json.dumps({"found": False, "message": str(e)}),
            properties=FunctionCallResultProperties(run_llm=False),
        )
        await params.llm.push_frame(TTSSpeakFrame(text=reply_text))
        return

    if not chunks:
        reply_text = _policy_unavailable_msg(lang)
        await params.result_callback(
            json.dumps({"found": False, "message": "No relevant policy content found."}),
            properties=FunctionCallResultProperties(run_llm=False),
        )
        await params.llm.push_frame(TTSSpeakFrame(text=reply_text))
        return

    context_text = "\n\n".join(chunks)

    # Build a language instruction for the RAG LLM. This secondary call has
    # no conversation history, so we must be explicit — the policy context is
    # always in English regardless of what language the user spoke.
    lang = lang_state.get("lang", "en")
    lang_instruction = (
        "Reply in Hindi (Devanagari script). The policy context may be in English "
        "but your answer must be in Hindi."
        if lang == "hi"
        else "Reply in English."
    )

    rag_prompt = (
        "You are FinBot, a voice assistant for IndiaFirst Bank. Answer the "
        "customer's question using ONLY the policy context below. Keep it to "
        "one short spoken sentence, plain language, no markdown, no symbols. "
        f"{lang_instruction} "
        "If the context doesn't actually answer the question, say you don't "
        "have that information and suggest contacting support at "
        "1800-200-1234.\n\n"
        f"Policy context:\n{context_text}\n\n"
        f"Customer question: {query}\n\n"
        "Answer:"
    )

    reply_text = await call_rag_llm(rag_prompt)

    logger.info(f"[latency] tool get_payment_policy (RAG, top-5): {(time.monotonic() - t0) * 1000:.0f} ms")

    await params.result_callback(
        json.dumps({"found": True, "answer": reply_text, "chunks_used": len(chunks)}),
        properties=FunctionCallResultProperties(run_llm=False),
    )
    await params.llm.push_frame(TTSSpeakFrame(text=reply_text))


payment_status_tool = FunctionSchema(
    name="get_payment_status",
    description="Get the payment status for a bank customer by first name.",
    properties={"name": {"type": "string", "description": "The customer's first name."}},
    required=["name"],
    handler=get_payment_status,
)

product_tool = FunctionSchema(
    name="get_product_recommendations",
    description=(
        "Find financial products from the FinBot product catalogue "
        "using deterministic product filters. Use this when the user "
        "is asking for a product recommendation."
    ),
    properties={
        "product_domain": {
            "type": "string",
            "description": (
                "Product domain. Use only: Banking, Lending, "
                "Investments, Insurance, Cards, Retirement."
            ),
        },
        "product_type": {
            "type": "string",
            "description": (
                "Specific product type such as SIP, Fixed Deposit, "
                "Personal Loan, Home Loan, Health Insurance, "
                "Travel Insurance, Credit Card, Savings Account, "
                "or Pension Plan."
            ),
        },
        "target_customer": {
            "type": "string",
            "description": "Target customer segment when clearly known.",
        },
        "region": {
            "type": "string",
            "description": "Customer region when relevant.",
        },
    },
    required=[],
    handler=get_product_recommendations,
)

balance_tool = FunctionSchema(
    name="get_balance",
    description="Get the account balance for a bank customer by first name.",
    properties={"name": {"type": "string", "description": "The customer's first name."}},
    required=["name"],
    handler=get_balance,
)

payment_policy_tool = FunctionSchema(
    name="get_payment_policy",
    description=(
        "Answer a customer question about IndiaFirst Bank's payment policy "
        "(retry limits, failure reasons, blocks, etc) using the bank's "
        "uploaded policy documents."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "The customer's policy question, in their own words.",
        }
    },
    required=["query"],
    handler=get_payment_policy,
)


# ================================================================
# PIPELINE — one instance built fresh per WebSocket connection
# ================================================================

async def run_bot(websocket: WebSocket):
    # Fire the LLM warm-up request immediately, in parallel with the
    # transport/STT/TTS setup below, so its cold-start cost is absorbed
    # before the user's real first turn rather than landing on it.
    asyncio.create_task(warm_up_llm())

    # ---- per-connection language state ----
    # Starts as English; LanguageTracker updates it on every final transcript.
    lang_state: dict = {"lang": "en"}

    # Thin closure wrappers so the module-level handlers can read lang_state.
    async def _product_handler(params):
        await get_product_recommendations(params, lang_state)

    async def _policy_handler(params):
        await get_payment_policy(params, lang_state)

    # Rebuild only the two slow tools with the closure handlers; the other two
    # (get_balance / get_payment_status) are fast and don't need per-connection
    # closures, so they continue to use the module-level FunctionSchema objects.
    _product_tool = FunctionSchema(
        name="get_product_recommendations",
        description=product_tool.description,
        properties=product_tool.properties,
        required=product_tool.required,
        handler=_product_handler,
    )
    _payment_policy_tool = FunctionSchema(
        name="get_payment_policy",
        description=payment_policy_tool.description,
        properties=payment_policy_tool.properties,
        required=payment_policy_tool.required,
        handler=_policy_handler,
    )

    transport = FastAPIWebsocketTransport(
    websocket=websocket,
    params=FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_sample_rate=AUDIO_IN_SAMPLE_RATE,
        audio_in_channels=1,
        audio_out_enabled=True,
        audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
        audio_out_channels=1,
        add_wav_header=False,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.2,
                min_volume=0.6,
                )
            ),  # <-- changed
        serializer=BrowserFrameSerializer(),
    ),
)

    stt = SarvamSTTService(
        api_key=SARVAM_API_KEY,
        model="saaras:v3",
        mode="transcribe",
        ttfs_p99_latency=0.2,
    )

    llm = OpenAILLMService(
        api_key=SARVAM_API_KEY,
        base_url="https://api.sarvam.ai/v1",
        settings=OpenAILLMService.Settings(
            model="sarvam-105b",
            extra={"reasoning_effort": None},
        ),
    )

    tts = SarvamTTSService(
        api_key=SARVAM_API_KEY,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            language="en-IN",
            voice="priya",
            pace=1.15,
            min_buffer_size=30,
            max_chunk_length=80,
        ),
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    context.set_tools([payment_status_tool, _product_tool, balance_tool, _payment_policy_tool])

    context_aggregator = LLMContextAggregatorPair(
    context=context,
    user_params=LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            stop=[
                SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=0.05,
                    wait_for_transcript=True,
                    )
                ]
            )
        ),
    )

    # --- latency instrumentation, one shared dict per connection ---
    # Anchored on the final STT transcript (see TurnStartProbe) rather than
    # VAD start/stop frames, which can fire more than once per utterance.
    latency_state: dict = {}

    vad_logger = VadEventLogger()
    turn_start_probe = TurnStartProbe(latency_state)
    transcript_tap = TranscriptTap()
    lang_tracker = LanguageTracker(lang_state)
    bot_text_tap = BotTextTap()

    llm_token_probe = LLMTokenProbe(latency_state)
    tts_first_audio_probe = TTSFirstAudioProbe(latency_state)

    bot_started_probe = LatencyStepProbe(
        latency_state,
        key="bot_started",
        frame_types=BotStartedSpeakingFrame,
        since_key="transcript_ready",
        label="END-TO-END (final transcript ready -> bot starts speaking)",
        emit=True,
    )

    metrics_logger = MetricsLogger()

    pipeline = Pipeline(
        [
            transport.input(),
            vad_logger,
            stt,
            turn_start_probe,
            transcript_tap,
            lang_tracker,
            context_aggregator.user(),
            llm,
            llm_token_probe,
            tts,
            bot_text_tap,
            tts_first_audio_probe,
            context_aggregator.assistant(),
            bot_started_probe,
            metrics_logger,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        logger.info("Browser client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ================================================================
# FASTAPI APP
# ================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload the embedding model once here so its "Loading
    # weights" cost is paid at boot, not live during a user's first
    # policy question (see modules/embedding.py preload_embedding_model).
    t0 = time.monotonic()
    await asyncio.to_thread(preload_embedding_model)
    logger.info(f"[startup] embedding model preloaded in {(time.monotonic() - t0) * 1000:.0f} ms")
    yield
    # Shutdown: nothing to clean up right now.


app = FastAPI(lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
POLICY_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Policy Data")
os.makedirs(POLICY_DATA_DIR, exist_ok=True)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/upload")
async def upload_policy(file: UploadFile = File(...)):
    """Hit by the "Update" button in the UI. Saves the uploaded PDF into
    ./Policy Data, then runs the full ingestion pipeline (extract -> chunk
    -> embed -> store in Chroma). Ingestion is CPU-bound (embedding model
    inference), so it's run in a worker thread via asyncio.to_thread to
    avoid blocking the event loop -- and, in particular, any in-progress
    voice calls being served by this same process."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

    save_path = os.path.join(POLICY_DATA_DIR, file.filename)

    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        num_chunks = await asyncio.to_thread(ingest_pdf, save_path)
    except Exception as e:
        logger.error(f"[upload] ingestion failed for {file.filename}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Ingestion failed: {e}"})

    logger.info(f"[upload] ingested {file.filename}: {num_chunks} chunks")
    return {"status": "ingested", "filename": file.filename, "chunks": num_chunks}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Browser client connected")
    try:
        await run_bot(websocket)
    except WebSocketDisconnect:
        logger.info("Browser client disconnected (WebSocketDisconnect)")


if __name__ == "__main__":
    print("=" * 60)
    print("  FinBot — IndiaFirst Bank Voice Agent (browser + pipecat)")
    print("  Latency-optimized: fast-path replies + tightened turn-stop.")
    print("  Policy answers are RAG-backed (upload PDFs via the Update button).")
    print("  Open http://localhost:8000 and tap the button to talk.")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
