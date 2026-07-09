
import os

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PIPELINE_DIR)


# =================== UDP / Networking ===================
HOST = "0.0.0.0"
PORT = 5005                              # receive Unity packets

UNITY_RESULT_PORT = 5006                 # Unity listens for VLM results on this port
UNITY_HOST_OVERRIDE = None               # None = auto from inbound traffic
VLM_PACKET_PREFIX = "VLM_RESULT"         # Unity-side filter / matching prefix


# =================== Display / Canvas ===================
LIVE_WINDOW = "Live Stream"          # header label drawn on the left pane
TARGET_WINDOW = "Target Result"      # header label drawn on the right pane
COMBINED_WINDOW = "MacProgram"       # single OS window that holds both panes
CANVAS_W, CANVAS_H = 1100, 1000
STREAM_W, STREAM_H = 1100, 1000
# Header bar height reserved at the top of each pane for the label -- keeps the
# label text out of the actual canvas so gaze/target overlays stay unobstructed.
PANE_HEADER_H = 32

POINT_RADIUS = 8
POINT_COLOR = (0, 0, 255)                # current gaze marker
TRAIL_COLOR = (0, 255, 255)              # in-progress gesture trail / gaze bbox
TARGET_COLOR = (0, 255, 0)               # YOLO target highlight
BG_COLOR = (30, 30, 30)


# =================== ADB Stream ===================
ADB_CMD = ["adb", "exec-out", "screenrecord", "--output-format=h264", "-"]


def build_ffmpeg_cmd(width: int, height: int):
    return [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "-",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}",
        "-",
    ]


# =================== Gaze / Gesture timing ===================
GAZE_BBOX_PADDING = 10
MIN_GAZE_POINTS_FOR_TARGET = 5
CAPTURE_DELAY_AFTER_END = 0.3

TARGET_SCORE_IOU_WEIGHT = 0.999
TARGET_MIN_OVERLAP = 0.03
TARGET_CROP_PAD_RATIO = 0.10             # 10% padding around target bbox for VLM crop

# Compare: number of objects to target, and a distinct overlay colour per slot.
COMPARE_TOP_N = 2
COMPARE_TARGET_COLORS = [(0, 255, 0), (0, 165, 255)]   # BGR: #1 green, #2 orange


# =================== Calibration (gaze_dir -> norm_xy and back) ===================
RIDGE_MODEL_PATH = "calibration_ridge_model.json"
RIDGE_MODEL_ABS_PATH = os.path.join(_PARENT_DIR, RIDGE_MODEL_PATH)


# =================== YOLO ===================
SEG_CONF = 0.15
SEG_IOU_THRESH = 0.50
SEG_MODEL_PATH = "yolov8n-seg.pt"
MAX_SEGMENTS_TO_RENDER = 80         # Cap drawn boxes only; scoring still uses all


# =================== CLIP + object DB ===================
# EVA-02-L/14 at 336px. Drop-in replacement for the older ViT-B/32; clearer
# score separation on the closed 3-object set. First load auto-downloads
# ~430 MB from HuggingFace.
CLIP_MODEL_NAME = "EVA02-L-14-336"
CLIP_PRETRAINED = "merged2b_s6b_b61k"

# Object DB lives in MacProgram/object_db/.
#   objects.json         metadata + per-object info text
#   images/<obj_id>/*    reference images for each object
#   embeddings.npz       cached CLIP embeddings, regenerated when images change
#                        OR when CLIP_MODEL_NAME/CLIP_PRETRAINED change.
OBJECT_DB_DIR = os.path.join(_PARENT_DIR, "object_db")
OBJECT_DB_JSON = os.path.join(OBJECT_DB_DIR, "objects.json")
OBJECT_DB_IMAGES_DIR = os.path.join(OBJECT_DB_DIR, "images")
OBJECT_DB_EMBEDDINGS_PATH = os.path.join(OBJECT_DB_DIR, "embeddings.npz")

# EVA-02 produces a tighter score distribution than ViT-B/32 -- matched pairs
# tend to land in 0.55-0.75 (with the embedding-space cosine being a bit lower
# than ViT-B/32 numerically). Keep masked_crop off per the reference paper.
CLIP_MATCH_MIN_SCORE = 0.55
CLIP_USE_MASKED_CROP = True


# =================== VLM (OpenAI GPT) ===================
OPENAI_MODEL = "gpt-5"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
VLM_OUTPUT_DIR = os.path.join(_PARENT_DIR, "vlm_outputs")
VLM_REQUEST_TIMEOUT_SEC = 30
VLM_MAX_OUTPUT_TOKENS = 2000
VLM_REASONING_EFFORT = "low"


# =================== OCR / Translate ===================
OCR_ROI_PAD_PX = 240
OCR_ROI_MIN_SIDE = 480
OCR_PARAGRAPH = True
OCR_MAX_SIDE = 960
OCR_MIN_CONF = 0.30


# =================== PROMPTS ===================

ASK_REFERENCE_PROMPT = """\
You are answering a user's open-ended question about a real-world object they pointed at in XR.
The image is a tight crop around the object. The object has already been identified for you
in a "DB info" block below -- treat that as authoritative ground truth and only fall back to
the image for things the DB does not cover (colour, condition, position, etc.). If the DB and
the image still aren't enough (e.g. the question is about specs, availability, recall status,
or anything outside the DB), you may use the web_search tool to fetch external info.

Output format (STRICT):
- Respond with PLAIN TEXT only -- the answer itself, nothing else.
- Do NOT wrap the answer in JSON, quotes, code fences, or any other envelope.
- Do NOT prefix with labels like "answer:" or the object name.
- The stream is rendered live in an XR card as tokens arrive, so any wrapper
  characters would appear in the UI.

Style:
- Natural conversational Korean, complete sentence(s).
- Concise -- one or two sentences is usually enough. Longer only when the
  user's question genuinely calls for detail.
- If the question can't be answered even with web search, say so plainly
  instead of guessing.
"""

# -------- Voice intent classifier --------
# The voice mode used to always call GPT with a free-form Q&A prompt. That
# meant a Voice user could never trigger the same DB-backed cards that
# gesture / UI users see (Search, Anchor, Save, Compare, Translate, Capture).
# This classifier picks the closest of the 7 canonical referents; anything
# unclear falls through to Ask, which is the open-ended fallback per the
# study design.
#
# The intent names MUST match the strings that Unity's ResultCardSpawner
# switches on (see ResultCardSpawner.HandleResult in Assets/Scripts/):
#   "Search/Find Info", "Ask", "Translate", "Compare", "Anchor",
#   "Save", "Capture".
VOICE_INTENT_PROMPT = """\
You classify a voice command spoken by a user wearing an XR headset. The
user is looking at a real-world object and may want to do one of the
following seven actions with it. Return EXACTLY the canonical intent name.

===== Canonical intents =====
1. "Search/Find Info" -- The user wants factual info / description of the
   object they are looking at. Typical phrasings: "what is this?", "tell me
   about this", "이거 뭐야?", "이게 뭔지 알려줘", "설명해줘".
2. "Translate" -- The user wants text visible on/near the target translated.
   Phrasings: "translate this", "이거 번역해줘", "read this in English".
3. "Compare" -- The user wants two visible objects compared. Phrasings:
   "compare these", "which one is better?", "이거랑 저거 비교해줘",
   "둘 중 뭐가 나아?".
4. "Anchor" -- The user wants a spatial anchor / pin dropped on the target
   so they can find it again later. Phrasings: "anchor this", "pin this
   here", "여기 표시해줘", "위치 저장해".
5. "Save" -- The user wants to attach a note / bookmark to the object.
   Phrasings: "save this", "note this", "이거 메모해줘", "북마크".
6. "Capture" -- The user wants to photograph / capture the target.
   Phrasings: "take a picture of this", "capture this", "찍어줘",
   "사진 저장".
7. "Ask" -- ANY open-ended question that does not clearly fit the six
   above. This is the fallback bucket. Phrasings: "how do I use this?",
   "how much is it?", "이거 어떻게 써?", "얼마야?", "누가 만들었어?".

===== Classification rules =====
- Prefer one of the first six intents when the phrasing clearly matches.
- If in doubt, choose "Ask". A study participant should never see a
  mis-routed card just because the intent classifier was overconfident.
- Answer in the SAME language the user spoke is NOT required here -- this
  is a machine-readable classification, so intent must be the exact
  canonical English string above.

===== Save-specific extraction =====
When and ONLY when the classified intent is "Save", also extract the note
body from the transcript into a `note_content` field. Users often speak
the note inline, e.g. "메모에 '내일 3시 회의' 라고 저장해줘" or
"save a note that says buy milk". The note body is the actual content the
user wants written on the sticky note -- NOT the wrapping command words.
Guidelines:
- Strip the wrapping verbs ("save", "note", "메모해줘", "저장해줘") and
  the framing quotes ("라고", "as", "that says").
- Preserve the note body in the user's original language.
- If the transcript is a Save intent but has no clear note body (e.g. the
  user just said "save this"), return an empty string for note_content --
  the downstream flow will fall back to opening the manual input UI.
- For every non-Save intent, omit `note_content` entirely (or leave it "").

===== Output format (STRICT) =====
Return EXACTLY one JSON object, no prose:

{
  "intent":       "<one of the seven canonical strings above>",
  "confidence":   "high" | "medium" | "low",
  "rationale":    "<brief English explanation, one short sentence>",
  "note_content": "<Save intent only: the extracted note body, else empty string>"
}

===== User transcript =====
"{transcript}"
"""


# DEPRECATED as of the 7-referent voice routing refactor. Voice queries now
# go through voice_pipeline.dispatch(), which classifies the transcript with
# VOICE_INTENT_PROMPT above and reuses the same YOLO+CLIP+DB handler as the
# gesture path. For open-ended Ask fallback the handler chain uses
# ASK_REFERENCE_PROMPT (see top of file).
#
# Kept here because it captures the GazePointAR Figure 8 structure with an
# explicit gaze coordinate injection, which is a useful reference if we ever
# want to run a single-call multimodal Voice mode again (e.g. for baseline
# comparison in a follow-up study). Not imported anywhere at runtime.
VOICE_COMMAND_PROMPT = """\
You are a context-aware voice assistant for a user wearing an XR headset. The
image below is a snapshot of the user's real-world field of view (passthrough
camera + Unity overlays) captured the moment their speech was recognized.
Treat pronouns in the query (this, that, here, there, it, they, 이것, 저것,
여기, 저기 등) as pointers to something visible in that image.

===== 1. User query (verbatim) =====
The user asked: "{transcript}"

===== 2. Where the user was LOOKING (gaze target) =====
{gaze_info}

===== 3. Where the user is POINTING (secondary cue) =====
Inspect the frame for a visible hand, extended finger, or hand-held pointer.
If one is present and clearly aimed at an object, treat that object as the
referent instead of the gaze target -- explicit pointing outranks gaze. If
no pointing gesture is visible, ignore this section and rely on the gaze
target above.

===== 4. Other objects in view (peripheral context) =====
Everything else visible in the frame (background objects, text, signage,
overlays) may still matter for questions like "what else is here?" or
"which of these ...?", but weight it below the gaze / pointing target when
resolving a specific referent.

===== 5. Answer this question =====
"{transcript}"

===== 6. Output format (STRICT) =====
Respond with EXACTLY one JSON object, no prose before or after:

{
  "name":       "<short label for the resolved target or task>",
  "referent":   "<what you resolved any pronoun to, grounded in the gaze/pointing target, e.g. 'the blue soda can at the gaze pixel'; empty string if the query had no pronoun>",
  "answer":     "<ONE natural sentence that directly answers the user, followed by a short justification clause>",
  "confidence": "high" | "medium" | "low"
}

===== 7. Answering rules =====
- Answer in the SAME language the user spoke (Korean transcript -> Korean
  answer; English transcript -> English answer).
- Keep the tone natural and conversational, like answering a curious friend.
- The "answer" field must be a single sentence. Include a short "because ..."
  or "-- <reason>" clause so the user understands why.
- Ground your answer in what is actually visible AT OR NEAR the gaze pixel
  first. Do NOT invent objects that are not in the frame.
- If the object at the gaze target is unclear, admit it in "answer", set
  "confidence" to "low", and describe what IS at that pixel neighbourhood so
  the user can tell whether the tracker mis-aimed.
- Even with missing info or an ambiguous referent, DO NOT refuse. Give your
  best estimate or a range and set "confidence" to "low".
- If the query has no pronoun and does not refer to the visible scene at
  all (e.g. a general knowledge question), still answer using the same
  format, leave "referent" empty, and set "confidence" based on how sure you
  are of the general answer.
"""
